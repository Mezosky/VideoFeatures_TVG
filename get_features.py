"""
Extract features for videos using pre-trained arquitectures

"""

import numpy as np
import pandas as pd
import logging
import torch
import os
import time

from tqdm import tqdm

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du

from models import build_model
from datasets import VideoSetDecord

import ipdb

# Logger
log = logging.getLogger(__name__)


def calculate_time_taken(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int((end_time - start_time) % 60)
    return hours, minutes, seconds


def create_csv(path, output_path, max_files="all"):
    assert (
        type(max_files) is not int or max_files != "all"
    ), "You must enter a int from the 1 to the N"

    if os.path.exists(output_path):
        proc_v = [v.split(".")[0] for v in os.listdir(output_path)]
        if len(proc_v) > 0:
            log.info(f"[Data] Already {len(proc_v)} files have been processed")
        entries = os.listdir(path)
        entries = [
            v.split(".")[0] for v in entries if v.split(".")[0] not in proc_v
        ]
    else:
        entries = os.listdir(path)
        entries = [v.split(".")[0] for v in entries]

    df = pd.DataFrame(entries)

    if max_files == "all":
        path_csv = path + "/videos_list.csv"
        if os.path.exists(path_csv):
            os.remove(path_csv)
        df.to_csv(path_csv, index=False, header=False)

    elif type(max_files) is int:
        indices = np.array_split(df.index, max_files)
        for i in range(max_files):
            path_csv = path + f"/videos_list_{i+1}.csv"
            if os.path.exists(path_csv):
                os.remove(path_csv)

            data_csv = df.loc[indices[i]]
            data_csv.to_csv(path_csv, index=False, header=False)


@torch.no_grad()
def perform_inference(test_loader, model, cfg):
    """
    Perform mutli-view testing that samples a segment of frames from a video
    and extract features from a pre-trained model.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    feat_arr = None
    for inputs in tqdm(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Perform the forward pass.
        preds, feat = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, feat = du.all_gather([preds, feat])

        feat = feat.cpu().numpy()

        if feat_arr is None:
            feat_arr = feat
        else:
            feat_arr = np.concatenate((feat_arr, feat), axis=0)
    return feat_arr


def test(cfg):
    """
    Perform multi-view testing/feature extraction on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # testing this
    import wrapt
    import tqdm.std

    methods = ["__del__", "close"]
    for method_name in methods:

        @wrapt.patch_function_wrapper(tqdm.std.tqdm, method_name)
        def new_del(wrapped, instance, args, kwargs):
            try:
                return wrapped(*args, **kwargs)
            except AttributeError:
                pass

    # testing this

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Comprobate and/or create ouput folder.
    try:
        folder_feature_name = cfg.TRAIN.CHECKPOINT_FILE_PATH.split("/")[
            -1
        ].split(".")[0]
    except:
        folder_feature_name = cfg.MODEL.ARCH
    output_path = os.path.join(cfg.OUTPUT_DIR, folder_feature_name)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        log.info(f"[Data] The directory {folder_feature_name} was created!")

    # Build the video model and print model statistics.
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)

    vid_root = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.DATA.PATH_PREFIX)

    # Check if the video list are all the videos or just a part of the list.
    if cfg.NUMBER_CSV == None:
        create_csv(cfg.DATA.PATH_TO_DATA_DIR, output_path)
        videos_list_file = os.path.join(
            cfg.DATA.PATH_TO_DATA_DIR, "videos_list.csv"
        )
    else:
        videos_list_file = os.path.join(
            cfg.DATA.PATH_TO_DATA_DIR, f"videos_list_{cfg.NUMBER_CSV}.csv"
        )

    log.info("[Data] Loading Video List...")
    with open(videos_list_file) as f:
        # Load the video's name
        videos = sorted(
            [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        )

        # Comprobate if some features were processed (new)
        if os.path.exists(output_path):
            proc_v = [v.split(".")[0] for v in os.listdir(output_path)]
            if len(proc_v) > 0:
                log.info(
                    f"[Data] Already {len(proc_v)} files have been processed"
                )
            videos = [
                v.split(".")[0]
                for v in videos
                if v.split(".")[0] not in proc_v
            ]

    log.info(f"[Data] {len(videos)} videos to be processed...")

    rejected_vids = []
    metadata_json_file = {}
    start_time = time.time()
    for vid_no, vid in enumerate(videos):

        # Create video testing loaders.
        path_to_vid = os.path.join(vid_root, os.path.split(vid)[0])
        vid_id = os.path.split(vid)[1]

        out_path = os.path.join(output_path, os.path.split(vid)[0])
        out_path_metadata = os.path.join(output_path, "metadata")
        out_file = vid_id.split(".")[0] + ".npy"

        log.info(f"[Model Inference] {vid_no + 1}.- Processing {vid}...")
        try:
            dataset = VideoSetDecord(cfg, path_to_vid, vid_id)
            log.info(
                f"[Model Inference]] {vid_no + 1}.- Video {vid} Processed."
            )
        except Exception as e:
            log.warning(
                f"[Model Inference] {vid_no + 1}.- Video {vid} cannot be read with error {e}."
            )
            rejected_vids.append(vid)
            continue

        test_loader = torch.utils.data.DataLoader(  # type: ignore
            dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            sampler=None,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
        )

        # Perform multi-view test on the entire dataset.
        feat_arr = perform_inference(test_loader, model, cfg)
        os.makedirs(out_path, exist_ok=True)

        if not feat_arr is None:
            np.save(os.path.join(out_path, out_file), feat_arr)

        dataset = None
        test_loader = None
        # feat_arr = None

    log.info("[Data] Rejected Videos: {}".format(rejected_vids))

    # * Execution Time
    end_time = time.time()
    hours, minutes, seconds = calculate_time_taken(start_time, end_time)
    log.info(
        f"Processed {len(videos)} videos with the model {cfg.MODEL.MODEL_NAME}, \
    it took a time of: {hours} hour(s), {minutes} minute(s) and {seconds} second(s)"
    )
