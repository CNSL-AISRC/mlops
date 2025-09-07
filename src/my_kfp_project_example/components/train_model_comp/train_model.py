from kfp import dsl
import os
import glob
from dotenv import load_dotenv

load_dotenv()


@dsl.component(
    base_image='pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime',
    target_image=f'{os.getenv("PRIVATE_DOCKER_REGISTRY")}/aasist-project/train-model:{os.getenv("TRAIN_MODEL_VERSION")}',
    packages_to_install=[
        'torchcontrib==0.0.2',
        #'numpy==2.3.1',
        'soundfile==0.13.1',
        'tensorboard==2.19.0',
        'tqdm==4.67.1',
        'mlflow==2.20',
        'boto3==1.37.1',
        'dotenv',
    ],
)
def train_model(processed_data: str, config: str) -> str:
    import argparse
    import json
    import os
    import random
    import sys
    import warnings
    from importlib import import_module
    from pathlib import Path
    from shutil import copy
    from typing import Dict, List, Union
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torchcontrib.optim import SWA
    from tqdm import tqdm
    import mlflow
    from datetime import datetime
    from data_utils import (Dataset_ASVspoof2019_train,
                            Dataset_ASVspoof2019_devNeval, genSpoof_list)
    from evaluation import calculate_tDCF_EER
    from utils import create_optimizer, seed_worker, set_seed, str_to_bool
    from mlflow.models import infer_signature
    def get_loader(
            database_path: str,
            seed: int,
            config: dict,
            data_ratio: float = 1.0) -> List[torch.utils.data.DataLoader]:
        """Make PyTorch DataLoaders for train / developement / evaluation"""
        track = config["track"]
        prefix_2019 = "ASVspoof2019.{}".format(track)

        trn_database_path = os.path.join(database_path, "ASVspoof2019_{}_train/".format(track))
        dev_database_path = os.path.join(database_path, "ASVspoof2019_{}_dev/".format(track))
        eval_database_path = os.path.join(database_path, "ASVspoof2019_{}_eval/".format(track))

        trn_list_path = os.path.join(database_path,
                        "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                            track, prefix_2019))
        dev_trial_path = os.path.join(database_path,
                        "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                            track, prefix_2019))
        eval_trial_path = os.path.join(
            database_path,
            "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
                track, prefix_2019))

        d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                                is_train=True,
                                                is_eval=False)
        
        # Subset training data if data_ratio < 1.0
        if data_ratio < 1.0:
            random.seed(seed)  # Ensure reproducible subsampling
            n_train_subset = int(len(file_train) * data_ratio)
            file_train = random.sample(file_train, n_train_subset)
            # Update labels dictionary to only include selected files
            d_label_trn = {k: v for k, v in d_label_trn.items() if k in file_train}
            print("no. training files (subset {:.1%}): {}".format(data_ratio, len(file_train)))
        else:
            print("no. training files:", len(file_train))

        train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                            labels=d_label_trn,
                                            base_dir=trn_database_path)
        gen = torch.Generator()
        gen.manual_seed(seed)
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen)

        _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                    is_train=False,
                                    is_eval=False)
        
        # Subset validation data if data_ratio < 1.0
        if data_ratio < 1.0:
            random.seed(seed)  # Ensure reproducible subsampling
            n_dev_subset = int(len(file_dev) * data_ratio)
            file_dev = random.sample(file_dev, n_dev_subset)
            print("no. validation files (subset {:.1%}): {}".format(data_ratio, len(file_dev)))
        else:
            print("no. validation files:", len(file_dev))

        dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                                base_dir=dev_database_path)
        dev_loader = DataLoader(dev_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

        file_eval = genSpoof_list(dir_meta=eval_trial_path,
                                is_train=False,
                                is_eval=True)
        print("no. evaluation files:", len(file_eval))
        eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                                base_dir=eval_database_path)
        eval_loader = DataLoader(eval_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

        return trn_loader, dev_loader, eval_loader

    def produce_evaluation_file(
        data_loader: DataLoader,
        model,
        device: torch.device,
        save_path: str,
        trial_path: str) -> None:
        """Perform evaluation and save the score to a file"""
        model.eval()
        with open(trial_path, "r") as f_trl:
            trial_lines = f_trl.readlines()
        fname_list = []
        score_list = []
        for batch_x, utt_id in tqdm(data_loader, desc="Evaluation Batches", leave=False):
            batch_x = batch_x.to(device)
            with torch.no_grad():
                _, batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

        # Create a mapping from trial lines for efficient lookup
        trial_dict = {}
        for trl in trial_lines:
            parts = trl.strip().split(' ')
            if len(parts) >= 5:
                _, utt_id, _, src, key = parts[:5]
                trial_dict[utt_id] = (src, key, trl)
        
        # Filter trial lines to match only the files we processed
        filtered_trial_data = []
        for fn in fname_list:
            if fn in trial_dict:
                filtered_trial_data.append(trial_dict[fn])
            else:
                raise ValueError(f"File {fn} not found in trial file {trial_path}")
        
        assert len(filtered_trial_data) == len(fname_list) == len(score_list)
        with open(save_path, "w") as fh:
            for fn, sco, (src, key, trl) in zip(fname_list, score_list, filtered_trial_data):
                fh.write("{} {} {} {}\n".format(fn, src, key, sco))
        print("Scores saved to {}".format(save_path))

    def train_epoch(
        trn_loader: DataLoader,
        model,
        optim: Union[torch.optim.SGD, torch.optim.Adam],
        device: torch.device,
        scheduler: torch.optim.lr_scheduler,
        config: argparse.Namespace):
        """Train the model for one epoch"""
        running_loss = 0
        num_total = 0.0
        ii = 0
        model.train()

        # set objective (Loss) functions
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        for batch_x, batch_y in tqdm(trn_loader, desc="Training Batches", leave=False):
            batch_size = batch_x.size(0)
            num_total += batch_size
            ii += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
            batch_loss = criterion(batch_out, batch_y)
            running_loss += batch_loss.item() * batch_size
            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
                scheduler.step()
            elif scheduler is None:
                pass
            else:
                raise ValueError("scheduler error, got:{}".format(scheduler))

        running_loss /= num_total
        return running_loss


    # Tmp config
    seed = 42
    data_ratio = 0.05 # 5% of data for training
    run_name = os.getenv("MLFLOW_RUN_NAME")
    # run_name should include timestamp
    run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("Run name: {}".format(run_name))
    
    # load experiment configurations
    with open(config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"
        
    set_seed(seed, config)
    def get_model(model_config: Dict, device: torch.device):
        """Define DNN model architecture"""
        module = import_module("models.{}".format(model_config["architecture"]))
        _model = getattr(module, "Model")
        model = _model(model_config).to(device)
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        print("no. model params:{}".format(nb_params))

        return model
    # define database related paths
    output_dir = "output"
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = os.path.join(processed_data, config["database_path"])
    dev_trial_path = os.path.join(database_path,
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = os.path.join(
        database_path,
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # define model related paths
    
    model_tag = "{}_{}_ep{}_bs{}_{}".format(
        track,
        model_config["architecture"],
        config["num_epochs"], config["batch_size"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    model_tag = os.path.join(output_dir, model_tag)
    model_save_path = os.path.join(model_tag, "weights")
    eval_score_path = os.path.join(model_tag, config["eval_output"])
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    #copy(args.config, model_tag / "config.conf")
    
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")
    
    # MLflow
    ## set tracking uri
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    ## print tracking uri
    print("Tracking URI: {}".format(mlflow.get_tracking_uri()))
    
    # set experiment
    mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    print("Experiment name: {}".format(mlflow_experiment_name))
    
    mlflow.set_experiment(mlflow_experiment_name)
    experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
    ## print experiment id
    print("Experiment ID: {}".format(experiment.experiment_id))
    
  
    # Ensure no active run before starting a new one
    if mlflow.active_run() is not None:
        mlflow.end_run()
    ## autolog
    mlflow.autolog()
    
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:
       
        ### log all config to mlflow
        
        mlflow.log_params(config)
        mlflow.log_params({"seed": seed, "data_ratio": data_ratio})
        model = get_model(model_config, device)
        
         # Include a signature of the model
        example_input = torch.randn(1, 64600, device=device) # 1 sample, 64600 features ~ 4 seconds of audio
        signature = infer_signature(example_input, model(example_input))
        
        mlflow.set_tag("dataset", "ASVspoof2019")
        mlflow.set_tag("track", track)
        mlflow.set_tag("model", model_config["architecture"])
        mlflow.set_tag("num_epochs", config["num_epochs"])
        mlflow.set_tag("batch_size", config["batch_size"])
        mlflow.set_tag("data_ratio", data_ratio)
        mlflow.set_tag("seed", seed)
        
        trn_loader, dev_loader, eval_loader = get_loader(
            database_path, seed, config, data_ratio)
        optim_config["steps_per_epoch"] = len(trn_loader)
        optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
        optimizer_swa = SWA(optimizer)

        best_dev_eer = 1.
        best_eval_eer = 100.
        best_dev_tdcf = 0.05
        best_eval_tdcf = 1.
        n_swa_update = 0  # number of snapshots of model to use in SWA
        best_model_path = None
        f_log = open(os.path.join(model_tag, "metric_log.txt"), "a")
        f_log.write("=" * 5 + "\n")

        # make directory for metric logging
        metric_path = os.path.join(model_tag, "metrics")
        os.makedirs(metric_path, exist_ok=True)

        # Training
        for epoch in tqdm(range(config["num_epochs"]), desc="Training Epochs"):
            print("Start training epoch{:03d}".format(epoch))
            running_loss = train_epoch(trn_loader, model, optimizer, device,
                                    scheduler, config)
            produce_evaluation_file(dev_loader, model, device,
                                    os.path.join(metric_path, "dev_score.txt"), dev_trial_path)
            dev_eer, dev_tdcf = calculate_tDCF_EER(
                cm_scores_file=os.path.join(metric_path, "dev_score.txt"),
                asv_score_file=os.path.join(database_path, config["asv_score_path"]),
                output_file=os.path.join(metric_path, "dev_t-DCF_EER_{}epo.txt".format(epoch)),
                printout=False)
            print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
                running_loss, dev_eer, dev_tdcf))
            
            writer.add_scalar("loss", running_loss, epoch)
            writer.add_scalar("dev_eer", dev_eer, epoch)
            writer.add_scalar("dev_tdcf", dev_tdcf, epoch)
            
            # log metrics to mlflow
            mlflow.log_metric("loss", running_loss, step=epoch)
            mlflow.log_metric("dev_eer", dev_eer, step=epoch)
            mlflow.log_metric("dev_tdcf", dev_tdcf, step=epoch)

            best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
            if best_dev_eer >= dev_eer:
                print("best model find at epoch", epoch)
                best_dev_eer = dev_eer
                best_model_path = os.path.join(model_save_path, "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))
                torch.save(model.state_dict(), best_model_path)
                # do evaluation whenever best model is renewed
                if str_to_bool(config["eval_all_best"]):
                    produce_evaluation_file(eval_loader, model, device,
                                            eval_score_path, eval_trial_path)
                    eval_eer, eval_tdcf = calculate_tDCF_EER(
                        cm_scores_file=eval_score_path,
                        asv_score_file=os.path.join(database_path, config["asv_score_path"]),
                        output_file=os.path.join(metric_path,
                        "t-DCF_EER_{:03d}epo.txt".format(epoch)))

                    log_text = "epoch{:03d}, ".format(epoch)
                    if eval_eer < best_eval_eer:
                        log_text += "best eer, {:.4f}%".format(eval_eer)
                        best_eval_eer = eval_eer
                    if eval_tdcf < best_eval_tdcf:
                        log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                        best_eval_tdcf = eval_tdcf
                        torch.save(model.state_dict(),
                                os.path.join(model_save_path, "best.pth"))
                    if len(log_text) > 0:
                        print(log_text)
                        f_log.write(log_text + "\n")

                print("Saving epoch {} for swa".format(epoch))
                optimizer_swa.update_swa()
                n_swa_update += 1
            writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
            writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)
            
            # log metrics to mlflow
            mlflow.log_metric("best_dev_eer", best_dev_eer, step=epoch)
            mlflow.log_metric("best_dev_tdcf", best_dev_tdcf, step=epoch)
        
       
        # TODO: Temporarily log last model to mlflow
        mlflow.log_artifact(output_dir)
        # mlflow.log_artifact(model_save_path)
        # mlflow.log_artifact(metric_path)
        #mlflow.log_artifact(eval_score_path)
       
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path="pytorch_last_model", registered_model_name=os.getenv("MLFLOW_MODEL_NAME"), signature=signature,  metadata={
        "input_description": "1D float32 tensor of raw audio waveform, sampled at 16kHz. Shape: [batch_size, 64600] (~4s audio)",
        "output_description": "Float32 tensor with shape [batch_size, 1], representing probability of deepfake voice."
        })
        
        # # convert to scripted model and log the model
        # print("Converting to scripted model")
        # #scripted_pytorch_model = torch.jit.script(model)
        # scripted_pytorch_model = torch.jit.trace(model, example_input)
        # print("Scripted model converted")
        

        # mlflow.pytorch.log_model(pytorch_model=scripted_pytorch_model, artifact_path="pytorch_last_model_scripted", registered_model_name=os.getenv("MLFLOW_MODEL_NAME"), signature=signature, metadata={
        # "input_description": "1D float32 tensor of raw audio waveform, sampled at 16kHz. Shape: [batch_size, 64600] (~4s audio)",
        # "output_description": "Float32 tensor with shape [batch_size, 1], representing probability of deepfake voice."
        # })
        
        
        model_uri = f"{run.info.artifact_uri}/pytorch_last_model"
        # Check inference
        # Load each saved model for inference

        loaded_model = mlflow.pytorch.load_model(model_uri)
        y_pred = loaded_model(example_input)
        print(f"predict X: {example_input}, y_pred: {y_pred}")
        print("Trained model:", model_uri)
        return model_uri
    #return "s3://mlflow/3/9a0242aedc534f5285b5edc4fa248731/artifacts/pytorch_last_model_scripted" # DUMMY OUTPUT FOR DEBBUGING
