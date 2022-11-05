#!/usr/bin/python
# -*- coding: utf-8 -*-

from audioop import reverse
import torch
from torch.functional import _return_counts
import torch.nn as nn
import torch.nn.functional as F
import numpy, sys, random
import time, itertools, importlib, os

from pathlib import Path

from tqdm import tqdm

from DatasetLoader import test_dataset_loader, loadWAV
from torch.cuda.amp import autocast, GradScaler

from utils import ReverseLayerF,  score_normalization


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None, label_dev=None, alpha=1.0):
        return self.module(x, label, label_dev, alpha)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, trainfunc_dev, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc_dev).__getattribute__("LossFunction")
        self.__Ldev__ = LossFunction(nOut=256, nClasses=4)#, **kwargs)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label, label_dev=None, alpha=1.0):

        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        else:

            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
            nloss, prec1 = self.__L__.forward(outp, label)
            
            if label_dev == None:
                return nloss, prec1
            else:
                reverse_feature = ReverseLayerF.apply(outp, alpha)
                reverse_nloss, reverse_prec1 = self.__Ldev__.forward(reverse_feature, label_dev)
                nloss = nloss + reverse_nloss

                return nloss, prec1, reverse_prec1

class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, epoch, n_epoch, verbose, multi_task=False):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        # EER or accuracy

        tstart = time.time()

        if multi_task:
            top1_dev = 0
            for data, data_label, data_label_dev in loader:
                #print(data_label_dev)
                data = data.transpose(1, 0)

                self.__model__.zero_grad()

                label = torch.LongTensor(data_label).cuda()
                label_dev = torch.LongTensor(data_label_dev).cuda()

                p = float(counter + epoch * len(loader)) / n_epoch / len(loader)
                alpha = 2. / (1. + numpy.exp(-10 * p)) - 1

                if self.mixedprec:
                    with autocast():
                        nloss, prec1, reverse_prec1 = self.__model__(data, label, label_dev, alpha)
                    self.scaler.scale(nloss).backward()
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update()
                else:
                    nloss, prec1, reverse_prec1 = self.__model__(data, label, label_dev, alpha)
                    nloss.backward()
                    self.__optimizer__.step()

                loss += nloss.detach().cpu().item()
                top1 += prec1.detach().cpu().item()
                top1_dev += reverse_prec1.detach().cpu().item()
                counter += 1
                index += stepsize

                telapsed = time.time() - tstart
                tstart = time.time()

                if verbose:
                    sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                    sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% TEER/TAcc_dev {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, top1_dev / counter, stepsize / telapsed))
                    sys.stdout.flush()

                if self.lr_step == "iteration":
                    self.__scheduler__.step()

            if self.lr_step == "epoch":
                self.__scheduler__.step()

            return (loss / counter, top1 / counter, top1_dev / counter)
        else:
            for data, data_label, _ in loader:
                #print(1)
                data = data.transpose(1, 0)

                self.__model__.zero_grad()

                label = torch.LongTensor(data_label).cuda()

                if self.mixedprec:
                    with autocast():
                        nloss, prec1 = self.__model__(data, label)
                    self.scaler.scale(nloss).backward()
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update()
                else:
                    nloss, prec1 = self.__model__(data, label)
                    nloss.backward()
                    self.__optimizer__.step()

                loss += nloss.detach().cpu().item()
                top1 += prec1.detach().cpu().item()
                counter += 1
                index += stepsize

                telapsed = time.time() - tstart
                tstart = time.time()

                if verbose:
                    sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                    sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, stepsize / telapsed))
                    sys.stdout.flush()

                if self.lr_step == "iteration":
                    self.__scheduler__.step()

            if self.lr_step == "epoch":
                self.__scheduler__.step()

            return (loss / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=100, num_eval=1, cohort_path=None, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Cohorts
        if cohort_path is None:
            cohorts = None
        else:
            cohorts = numpy.load(cohort_path)

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        # HARD CODE -------------------------------------------------------- HARD CODE !!!!
        #lines = lines[:100]
        public_path = '/content/drive/MyDrive/VLSP2022/extracted_dataset/imsv-public-test'
        enrol_path = '/content/drive/MyDrive/VLSP2022/dataset/I-MSV-DATA'
        for idx, line in enumerate(lines):
          data = line.strip().split(',')
          data[1] = os.path.join(public_path, data[1])
          data[2] = os.path.join(enrol_path, data[2])
          lines[idx] = ','.join(data)
          
        # HARD CODE -------------------------------------------------------- HARD CODE !!!!

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split(',')[-3:-1] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_path_ = '' # HARD CODE !!!!
        test_dataset = test_dataset_loader(setfiles, test_path_, num_eval=num_eval, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        for idx, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu()
    
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split(',')

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                # NOTE: distance for training, normalized score for evaluating and testing
                if cohort_path is None:
                    #score = torch.inner(ref_feat.reshape(-1), com_feat.reshape(-1)).detach().cpu().numpy()
                    # dist = F.pairwise_distance(
                    # ref_feat.reshape(num_eval, -1),
                    # com_feat.reshape(num_eval, -1)).detach().cpu().numpy()
                    # score = -1 * numpy.mean(dist)
                    dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()
                    score = -1 * numpy.mean(dist)
                else:
                    score = score_normalization(ref_feat,
                                                com_feat,
                                                cohorts,
                                                top=100)

                all_scores.append(score)
                all_labels.append(int(data[-1]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()
        
        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Prepare cohorts
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def prepare(self,
                from_path='../data/test',
                save_path='checkpoints',
                prepare_type='cohorts',
                num_eval=1,
                eval_frames=0,
                print_interval=1):
        """
        Prepared 1 of the 2:
        1. Mean L2-normalized embeddings for known speakers.
        2. Cohorts for score normalization.
        """
        # HARD CODE -------------------------------------------------------- HARD CODE !!!!
        org_root_path = '/content/drive/MyDrive/VLSP2022/dataset/I-MSV-DATA'
        # HARD CODE -------------------------------------------------------- HARD CODE !!!!

        tstart = time.time()
        self.__model__.eval()
        if prepare_type == 'cohorts':
            # Prepare cohorts for score normalization.
            feats = []
            read_file = from_path
            files = []
            used_speakers = []
            with open(read_file) as listfile:
                lines = listfile.readlines()
                # Skip header.
                lines = lines[1:]
                for line in lines:
                    data = line.strip().split(',')
                    data_class = data[0]
                    utterance_path = data[2]
                    if data_class not in used_speakers:
                        used_speakers.append(data_class)
                        files.append(os.path.join(org_root_path, utterance_path))
                # while True:
                #     line = listfile.readline()
                #     if (not line):
                #         break
                #     data = line.split()

                #     data_1_class = Path(data[1]).parent.stem
                #     data_2_class = Path(data[2]).parent.stem
                #     if data_1_class not in used_speakers:
                #         used_speakers.append(data_1_class)
                #         files.append(data[1])
                #     if data_2_class not in used_speakers:
                #         used_speakers.append(data_2_class)
                #         files.append(data[2])
            setfiles = list(set(files))
            setfiles.sort()

            # Save all features to file
            for idx, f in enumerate(tqdm(setfiles)):
                inp1 = torch.FloatTensor(
                    loadWAV(f, eval_frames, evalmode=True,
                            num_eval=num_eval)).cuda()

                feat = self.__model__.module.__S__.forward(inp1)
                if self.__model__.module.__L__.test_normalize:
                    feat = F.normalize(feat, p=2,
                                       dim=1).detach().cpu().numpy().squeeze()
                else:
                    feat = feat.detach().cpu().numpy().squeeze()
                feats.append(feat)

            filename = 'cohort.npy'
            numpy.save(os.path.join(save_path, filename), numpy.array(feats))
        elif prepare_type == 'embed':
            # Prepare mean L2-normalized embeddings for known speakers.
            speaker_dirs = [x for x in Path(from_path).iterdir() if x.is_dir()]
            embeds = None
            classes = {}
            # Save mean features
            for idx, speaker_dir in enumerate(speaker_dirs):
                classes[idx] = speaker_dir.stem
                files = list(speaker_dir.glob('*.wav'))
                mean_embed = None
                for f in files:
                    embed = self.embed_utterance(
                        f,
                        eval_frames=eval_frames,
                        num_eval=num_eval,
                        normalize=self.__L__.test_normalize)
                    if mean_embed is None:
                        mean_embed = embed.unsqueeze(0)
                    else:
                        mean_embed = torch.cat(
                            (mean_embed, embed.unsqueeze(0)), 0)
                mean_embed = torch.mean(mean_embed, dim=0)
                if embeds is None:
                    embeds = mean_embed.unsqueeze(-1)
                else:
                    embeds = torch.cat((embeds, mean_embed.unsqueeze(-1)), -1)
                telapsed = time.time() - tstart
                if idx % print_interval == 0:
                    sys.stdout.write(
                        "\rReading %d of %d: %.4f s, embedding size %d" %
                        (idx, len(speaker_dirs), telapsed / (idx + 1), embed.size()[1]))
            print('')
            print(embeds.shape)
            # embeds = rearrange(embeds, 'n_class n_sam feat -> n_sam feat n_class')
            torch.save(embeds, Path(save_path, 'embeds.pt'))
            numpy.save(Path(save_path, 'classes.npy'), classes)
        else:
            raise NotImplementedError
    
    def get_embedding(self,
                        filename='embeddings.npy',
                        from_path='../data/test',
                        save_path='checkpoints',
                        num_eval=1,
                        eval_frames=0):
        org_root_path = '/content/drive/MyDrive/VLSP2022/extracted_dataset'
        self.__model__.eval()

        feats = []
        read_file = from_path
        files = []
        with open(read_file) as listfile:
            lines = listfile.readlines()
            
            for line in lines:
                data = line.strip()
                files.append(os.path.join(org_root_path, data))

            # Save all features to file
        for idx, f in enumerate(tqdm(files)):
            inp1 = torch.FloatTensor(
                loadWAV(f, eval_frames, evalmode=True,
                        num_eval=num_eval)).cuda()

            feat = self.__model__.module.__S__.forward(inp1)
            if self.__model__.module.__L__.test_normalize:
                feat = F.normalize(feat, p=2,
                                    dim=1).detach().cpu().numpy().squeeze()
            else:
                feat = feat.detach().cpu().numpy().squeeze()
            feats.append(feat)

        numpy.save(os.path.join(save_path, filename), numpy.array(feats))
