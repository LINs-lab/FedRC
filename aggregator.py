import os
import time
import random

from abc import ABC, abstractmethod
from copy import copy, deepcopy

import numpy as np
import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

from utils.torch_utils import *
import torch.nn as nn
# from utils.utils import get_learner
# from utils.utils import add_new_learner

def add_new_learner(learner, split):
    if split:
        new_learner = deepcopy(learner)
        new_learner.model = nn.Linear(new_learner.model.in_features, new_learner.model.output_dim)
    else:
        new_learner = deepcopy(learner)
        new_learner.model.classifier[1] = nn.Linear(new_learner.model.classifier[1].in_features, 10)

    return new_learner


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients: List[Client]

    test_clients: List[Client]

    global_learners_ensemble: List[Learner]

    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    n_clients:

    n_learners:

    clients_weights:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    global_train_logger:

    global_test_logger:

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__
    mix

    update_clients

    update_test_clients

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False,
            *args,
            **kwargs
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)
        self.method = method
        self.experiment = experiment
        self.suffix = suffix
        self.split = split
        self.domain_disc=domain_disc

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.global_learners_ensemble = global_learners_ensemble
        self.device = self.global_learners_ensemble.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger

        self.model_dim = self.global_learners_ensemble.model_dim
        self.prototype_dim = self.global_learners_ensemble.prototype_dim

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)
        self.n_learners = len(self.global_learners_ensemble)

        self.clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.c_round = 0
        self.write_logs()

    def remove_learner(self, index):
        self.n_learners -= 1
        self.global_learners_ensemble.remove_learner(index)

    def add_learner(self, index):
        new_learner = self.global_learners_ensemble.learners[index]
        # average_learners(self.global_learners_ensemble.learners, new_learner)
        new_learner = add_new_learner(new_learner, self.split)
        self.n_learners += 1
        self.global_learners_ensemble.add_learner(index)
        

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def update_test_clients(self):
        for client in self.test_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)
            if self.split:
                copy_model(target=client.learners_ensemble.base_learner.model, source=self.global_learners_ensemble.base_learner.model)

        for client in self.test_clients:
            client.update_sample_weights()
            client.update_learners_weights()
            # print(client.samples_weights)
            # client.update_learners_weights_single()


    def write_logs(self):
        self.update_test_clients()
        test_train_acces = []
        test_test_acces = []

        for global_logger, clients, client_type in [
            (self.global_train_logger, self.clients, 'train'),
            (self.global_test_logger, self.test_clients, 'test')
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if client_type == 'test':
                    test_train_acces.append(train_acc)
                    test_test_acces.append(test_acc)

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")

                    with np.printoptions(precision=3, suppress=True):
                        print("Pi: ", client.learners_weights.numpy())

                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)
                with open('./logs/{}/results-{}-{}.txt'.format(self.experiment, self.method, self.suffix), 'a+') as f:
                    f.write('{}, {}, {}, {}\n'.format(global_train_loss, global_train_acc, global_test_loss, global_test_acc))

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        print("+" * 30)
        
        print('test train accs: ' + str(test_train_acces))
        print('test_test_acces: ' + str(test_test_acces))
        print('test train variance: ' + str(torch.std(torch.tensor(test_train_acces) + 0.0).item()))
        print('test test variance: ' + str(torch.std(torch.tensor(test_test_acces) + 0.0).item()))
        print('test train mean: ' + str(torch.mean(torch.tensor(test_train_acces) + 0.0).item()))
        print('test test mean: ' + str(torch.mean(torch.tensor(test_test_acces) + 0.0).item()))
        with open('./logs/{}/test-results-{}-{}.txt'.format(self.experiment, self.method, self.suffix), 'a+') as f:
            f.write('test train accs: ' + str(test_train_acces) + '\n')
            f.write('test_test_acces: ' + str(test_test_acces) + '\n')
            f.write('test train variance: ' + str(torch.std(torch.tensor(test_train_acces) + 0.0).item()) + '\n')
            f.write('test test variance: ' + str(torch.std(torch.tensor(test_test_acces) + 0.0).item()) + '\n')
            f.write('test train mean: ' + str(torch.mean(torch.tensor(test_train_acces) + 0.0).item()) + '\n')
            f.write('test test mean: ' + str(torch.mean(torch.tensor(test_test_acces) + 0.0).item()) + '\n')

        if self.verbose > 0:
            print("#" * 80)


    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            save_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            torch.save(learner.model.state_dict(), save_path)

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            save_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            for client_id, client in enumerate(clients):
                weights[client_id] = client.learners_ensemble.learners_weights

            np.save(save_path, weights)

    def load_state(self, dir_path):
        """
        load the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         from a `.pt` file, and `learners_weights` for each client in `self.clients` from numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            chkpts_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            learner.model.load_state_dict(torch.load(chkpts_path))

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            chkpts_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            weights = np.load(chkpts_path)

            for client_id, client in enumerate(clients):
                client.learners_ensemble.learners_weights = weights[client_id]

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        clients_indicies = [i for i in range(len(self.clients))]
        if self.sample_with_replacement:
            sampled_indices = \
                self.rng.choices(
                    population=clients_indicies,
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            sampled_indices= self.rng.sample(clients_indicies, k=self.n_clients_per_round)
        
        self.sampled_clients = [self.clients[i] for i in sampled_indices]

        return sampled_indices

    def sample_clients_group(self, clients_indicies):
        if len(clients_indicies) == 0:
            return []
        if self.sample_with_replacement:
            sampled_indices = \
                self.rng.choices(
                    population=clients_indicies,
                    weights=self.clients_weights,
                    k=int(self.sampling_rate * len(clients_indicies))
                )
        else:
            sampled_indices= self.rng.sample(clients_indicies, k=int(self.sampling_rate * len(clients_indicies)))
        
        self.sampled_clients = [self.clients[i] for i in sampled_indices]

        return sampled_indices


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self, diverse=False):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=self.clients_weights)

        self.c_round += 1
        self.update_clients()

        if self.c_round % self.log_freq == 0:
            self.write_logs()


    def update_clients(self):
        for client in self.test_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)
            if self.split:
                copy_model(target=client.learners_ensemble.base_learner.model, source=self.global_learners_ensemble.base_learner.model)


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """
    def mix(self, diverse=True):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step(diverse=diverse)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]
            
            sampled_weights = torch.tensor([client.n_train_samples for client in self.sampled_clients])
            average_learners(learners, learner, weights=sampled_weights / sum(sampled_weights))
        
        if self.split:
            base_learners = [client.learners_ensemble.base_learner for client in self.sampled_clients]
            average_learners(base_learners, self.global_learners_ensemble.base_learner, weights=torch.ones((len(self.sampled_clients),)) / len(self.sampled_clients))

        if self.domain_disc:
            disc_learners = [client.learners_ensemble.domain_disc_learner for client in self.sampled_clients]
            average_learners(disc_learners, self.global_learners_ensemble.domain_disc_learner, weights=torch.ones((len(self.sampled_clients),)) / len(self.sampled_clients))

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(learner.model, self.global_learners_ensemble[learner_id].model)

                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )
            if self.split:
                copy_model(target=client.learners_ensemble.base_learner.model, source=self.global_learners_ensemble.base_learner.model)

            if self.domain_disc:
                copy_model(target=client.learners_ensemble.domain_disc_learner.model, source=self.global_learners_ensemble.domain_disc_learner.model)

    

class IFCAAggregator(CentralizedAggregator):
    def mix(self, diverse=False):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step(diverse=diverse)


        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients if client.learners_ensemble.learners_weights[learner_id] > 0.9]
            # get weights
            learner_clients = [client for client in self.sampled_clients if client.learners_ensemble.learners_weights[learner_id] > 0.9]
            sampled_clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in learner_clients],
                dtype=torch.float32
            )
            sampled_clients_weights = sampled_clients_weights / sampled_clients_weights.sum()

            # learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=sampled_clients_weights)
        
        if self.split:
            base_learners = [client.learners_ensemble.base_learner for client in self.sampled_clients]
            average_learners(base_learners, self.global_learners_ensemble.base_learner, weights=sampled_clients_weights)

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class FeSEMAggregator(CentralizedAggregator):

    def mix(self, diverse=False):
        self.sample_clients()


        for client in self.sampled_clients:
            client.step(diverse=diverse)
            target_learner = torch.nonzero(client.learners_ensemble.learners_weights == max(client.learners_ensemble.learners_weights))[0]
            if client.learners_ensemble.learners_weights[target_learner] > 0.9:
                distances = torch.tensor([get_learner_distance(learner, client.learners_ensemble[target_learner]) for learner in self.global_learners_ensemble])
                client.distances = distances
            client.update_sample_weights()
            client.update_learners_weights()
            # client.step(diverse=diverse)


        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients if client.learners_ensemble.learners_weights[learner_id] > 0.9]
            # get weights
            learner_clients = [client for client in self.sampled_clients if client.learners_ensemble.learners_weights[learner_id] > 0.9]
            sampled_clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in learner_clients],
                dtype=torch.float32
            )
            sampled_clients_weights = sampled_clients_weights / sampled_clients_weights.sum()

            # learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=sampled_clients_weights)
        
        if self.split:
            base_learners = [client.learners_ensemble.base_learner for client in self.sampled_clients]
            average_learners(base_learners, self.global_learners_ensemble.base_learner, weights=sampled_clients_weights)

        if self.domain_disc:
            disc_learners = [client.learners_ensemble.domain_disc_learner for client in self.sampled_clients]
            average_learners(disc_learners, self.global_learners_ensemble.domain_disc_learner, weights=torch.ones((len(self.sampled_clients),)) / len(self.sampled_clients))

        self.update_clients()

        # assign the updated model to all clients
        # self.update_clients()

        # for client in self.sampled_clients:
        #     client.step(diverse=diverse)


        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class PersonalizedAggregator(CentralizedAggregator):
    r"""
    Clients do not synchronize there models, instead they only synchronize optimizers, when needed.

    """
    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(self.global_learners_ensemble[learner_id].model.parameters())
            if self.split:
                if callable(getattr(client.learners_ensemble.base_learner.optimizer, "set_initial_params", None)):
                        client.learners_ensemble.base_learner.optimizer.set_initial_params(self.global_learners_ensemble.base_learner.model.parameters())
            if self.domain_disc:
                if callable(getattr(client.learners_ensemble.base_learner.optimizer, "set_initial_params", None)):
                        client.learners_ensemble.domain_disc_learner.optimizer.set_initial_params(self.global_learners_ensemble.domain_disc_learner.model.parameters())


class APFLAggregator(Aggregator):
    r"""
    Implements
        `Adaptive Personalized Federated Learning`__(https://arxiv.org/abs/2003.13461)

    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            alpha,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False
    ):
        super(APFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
        assert self.n_learners == 2, "APFL requires two learners"

        self.alpha = alpha

    def mix(self, diverse=False):
        self.sample_clients()

        for client in self.sampled_clients:
            for _ in range(client.local_steps):
                # client.step(single_batch_flag=True)
                client.step()

                partial_average(
                    learners=[client.learners_ensemble[1]],
                    average_learner=client.learners_ensemble[0],
                    alpha=self.alpha
                )

        average_learners(
            learners=[client.learners_ensemble[0] for client in self.clients],
            target_learner=self.global_learners_ensemble[0],
            weights=self.clients_weights
        )
        if self.split:
            average_learners(
                learners=[client.learners_ensemble.base_learner for client in self.clients],
                target_learner=self.global_learners_ensemble.base_learner,
                weights=self.clients_weights
            )
        if self.domain_disc:
            disc_learners = [client.learners_ensemble.domain_disc_learner for client in self.sampled_clients]
            average_learners(disc_learners, self.global_learners_ensemble.domain_disc_learner, weights=self.clients_weights)


        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client in self.clients:

            copy_model(client.learners_ensemble[0].model, self.global_learners_ensemble[0].model)

            if self.split:
                copy_model(client.learners_ensemble.base_learner.model, self.global_learners_ensemble.base_learner.model)

            if callable(getattr(client.learners_ensemble[0].optimizer, "set_initial_params", None)):
                client.learners_ensemble[0].optimizer.set_initial_params(
                    self.global_learners_ensemble[0].model.parameters()
                )
        
        for client in self.test_clients:
            copy_model(client.learners_ensemble[0].model, self.global_learners_ensemble[0].model)
            copy_model(client.learners_ensemble[1].model, self.global_learners_ensemble[0].model)
            if self.split:
                copy_model(client.learners_ensemble.base_learner.model, self.global_learners_ensemble.base_learner.model)


class LoopLessLocalSGDAggregator(PersonalizedAggregator):
    """
    Implements L2SGD introduced in
    'Federated Learning of a Mixture of Global and Local Models'__. (https://arxiv.org/pdf/2002.05516.pdf)


    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            communication_probability,
            penalty_parameter,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False
    ):
        super(LoopLessLocalSGDAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )

        self.communication_probability = communication_probability
        self.penalty_parameter = penalty_parameter

    @property
    def communication_probability(self):
        return self.__communication_probability

    @communication_probability.setter
    def communication_probability(self, communication_probability):
        self.__communication_probability = communication_probability

    def mix(self):
        communication_flag = self.np_rng.binomial(1, self.communication_probability, 1)

        if communication_flag:
            for learner_id, learner in enumerate(self.global_learners_ensemble):
                learners = [client.learners_ensemble[learner_id] for client in self.clients]
                average_learners(learners, learner, weights=self.clients_weights)

                partial_average(
                    learners,
                    average_learner=learner,
                    alpha=self.penalty_parameter/self.communication_probability
                )

                self.update_clients()

                self.c_round += 1

                if self.c_round % self.log_freq == 0:
                    self.write_logs()

        else:
            self.sample_clients()
            for client in self.sampled_clients:
                client.step(single_batch_flag=True)



class FedIASAggregator(CentralizedAggregator):

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            tol=0.05,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False
    ):

        super(FedIASAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )


        self.tol = tol
        self.clients_updates = {}
        self.clients_cluster_indicies = {}
        self.similarities = np.zeros((self.n_clients, self.n_clients))

    def cos_sim(self, x, y):
        sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return 1 - sim

    def proto_sim_mean(self, x, y):
        sim = 0.0
        count = 0
        for i in range(len(x)):
            if np.sum(x[i]) == 0 or np.sum(y[i]) == 0:
                continue
            temp_sim = self.cos_sim(x[i], y[i])
            sim = sim + temp_sim
            count = count + 1
        # return 1 - sim
        return sim / count if count > 0 else sim

    def proto_sim(self, x, y):
        sim = 0.0
        for i in range(len(x)):
            if np.sum(x[i]) == 0 or np.sum(y[i]) == 0:
                continue
            temp_sim = self.cos_sim(x[i], y[i])
            if temp_sim > sim:
                sim = temp_sim
        # return 1 - sim
        return sim

    def proto_sim_cs(self, x, y):
        sim = 0.0
        for i in range(len(x) - 1):
            if np.sum(x[i]) == 0 or np.sum(y[i]) == 0:
                continue
            temp_sim = self.cos_sim(x[i], y[i])
            if temp_sim > sim:
                sim = temp_sim
        # return 1 - sim
        return sim

    def feature_norm_sim(self, x, y):
        return torch.norm(x - y) / len(x)
    
    def save_similarites(self, epoch_num):
        file = './logs/{}/sims-{}-{}-{}.txt'.format(self.experiment, self.method, self.suffix, epoch_num)
        with open(file, 'w') as f:
            for i in range(len(self.clients)):
                client_label_status = [0 for _ in range(self.clients[0].class_number)]
                for k, v in self.clients[i].label_stats.items():
                    client_label_status[k] = v
                f.write('{}\t{}\t{}\t{}\n'.format(self.clients[i].data_type, self.clients[i].feature_types[0],client_label_status, self.clients[i].learners_ensemble.learners_weights))
            for i in range(len(self.clients)):
                for j in range(len(self.clients)):
                    f.write('{}\t'.format(self.similarities[i][j]))
                f.write('\n')
    
    def mix(self, diverse=False):
        sampled_indicies = self.sample_clients()
        alpha = self.tol
        # C = self.global_learners_ensemble[0].model.output_dim
        # cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

        
        # # Gradients: get client updates
        # for client_id in sampled_indicies:
        #     client = self.clients[client_id]
        #     client_update_full = client.step()
        #     client_cluster_index = torch.nonzero(client.learners_ensemble.learners_weights==torch.max(client.learners_ensemble.learners_weights)).squeeze()

        # self.clients_cluster_indicies[client_id] = int(client_cluster_index)
        # self.clients_updates[client_id] = client_update_full[client_cluster_index,:]

        # # Gradients: get the similarity of chosen clients, only clients belongs to the same cluster will have the similarity, others will set to 1.0
        # for index_i, client_i in enumerate(self.clients):
        #     for index_j, client_j in enumerate(self.clients):
        #         self.similarities[index_i][index_j] = self.cos_sim(self.clients_updates[index_i],self.clients_updates[index_j])

        # Prototypes FeSEM: get client updates for FeSEM
        # for client_id in sampled_indicies:
        #     client = self.clients[client_id]
        #     # print(client_cluster_index, client.learners_ensemble.learners_weights[client_cluster_index])
            
        #     if max(client.learners_ensemble.learners_weights) < 1.0:
        #         client.update_sample_weights()
        #         client.update_learners_weights()
        #     client_cluster_index = torch.nonzero(client.learners_ensemble.learners_weights == max(client.learners_ensemble.learners_weights))[0]

        #     client_update_full = client.step(diverse=diverse)

        #     distances = torch.tensor([get_learner_distance(learner, client.learners_ensemble[client_cluster_index]) for learner in self.global_learners_ensemble])
        #     client.distances = distances
        #     client.update_sample_weights()
        #     client.update_learners_weights()

        #     client_cluster_index = torch.nonzero(client.learners_ensemble.learners_weights == max(client.learners_ensemble.learners_weights))[0]

        #     # print(distances, client_cluster_index)

        #     self.clients_cluster_indicies[client_id] = int(client_cluster_index)
        #     self.clients_updates[client_id] = client_update_full
        
        # Prototypes: get client updates
        for client_id in sampled_indicies:
            client = self.clients[client_id]
            client_update_full = client.step()
            client_cluster_index = torch.nonzero(client.learners_ensemble.learners_weights==torch.max(client.learners_ensemble.learners_weights)).squeeze()
            if len(client_cluster_index.shape) >= 1:
                client_cluster_index = client_cluster_index[0]
            # if client_cluster_index.shape[0] > 1:
            #     client_cluster_index = client_cluster_index[0]

            self.clients_cluster_indicies[client_id] = int(client_cluster_index)
            self.clients_updates[client_id] = client_update_full

        # Prototypes: get the similarity of chosen clients.
        for i, client_i in enumerate(sampled_indicies):
            for j, client_j in enumerate(sampled_indicies):
                if client_i in self.clients_updates and client_j in self.clients_updates:
                    # self.similarities[client_i][client_j] = self.proto_sim(self.clients_updates[client_i],self.clients_updates[client_j])
                    self.similarities[client_i][client_j] = self.proto_sim_cs(self.clients_updates[client_i],self.clients_updates[client_j])
                    # self.similarities[client_i][client_j] = self.proto_sim_mean(self.clients_updates[client_i],self.clients_updates[client_j])


        # update global learners
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]
            
            sampled_weights = torch.tensor([client.n_train_samples for client in self.sampled_clients])
            average_learners(learners, learner, weights=sampled_weights / sum(sampled_weights))
        
        # split clients by clusters, then choosen the cluster with minimum similarity.
        min_clients_by_clusters = None
        min_similarities_by_clusters = None
        min_sim_n_clients_by_clusters = None
        min_sim_cluster_index = 0
        min_sim = 0.0
        decide_removes = []
        for cluster_index in range(self.n_learners):
            # print(self.clients_updates, self.clients_cluster_indicies, cluster_index[0])
            clients_with_cluster_index = [(client, i) for i, client in enumerate(self.clients) if (i in self.clients_updates and i in self.clients_cluster_indicies and self.clients_cluster_indicies[i] == cluster_index)]
            if len(clients_with_cluster_index) == 0:
                decide_removes.append(cluster_index)
                continue
            n_clients_with_cluster_index = len(clients_with_cluster_index)
            similarities_with_that_cluster = np.zeros((n_clients_with_cluster_index, n_clients_with_cluster_index))
            for i in range(n_clients_with_cluster_index):
                for j in range(n_clients_with_cluster_index):
                    # ablation study 1
                    # similarities_with_that_cluster[i][j] = self.similarities[ clients_with_cluster_index[i][1]][clients_with_cluster_index[j][1]]                   
                    # full version
                    similarities_with_that_cluster[i][j] = self.similarities[ clients_with_cluster_index[i][1]][clients_with_cluster_index[j][1]] * clients_with_cluster_index[i][0].learners_ensemble.learners_weights[cluster_index] * clients_with_cluster_index[j][0].learners_ensemble.learners_weights[cluster_index]
            # choose the cluster with the least mean similarity
            # mean_sim_with_that_cluster = np.max(similarities_with_that_cluster)
            max_sim_with_that_cluster = np.max(similarities_with_that_cluster)
            if max_sim_with_that_cluster > min_sim:
                # min_sim = mean_sim_with_that_cluster
                min_sim = max_sim_with_that_cluster
                min_clients_by_clusters = clients_with_cluster_index
                min_similarities_by_clusters = similarities_with_that_cluster
                min_sim_n_clients_by_clusters = n_clients_with_cluster_index
                min_sim_cluster_index = cluster_index
            



        # Get all sampled clients belong to that cluster, group them into two clusters based on the similarity, then replace the old cluster by two new clusters.  
        # print(np.min(min_similarities_by_clusters))
        print(min_sim, np.mean(min_similarities_by_clusters), np.std(min_similarities_by_clusters), min_sim_cluster_index)

        if min_sim - np.mean(min_similarities_by_clusters) > alpha:
            # split these clients into two clusters
            clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete")
            clustering.fit(min_similarities_by_clusters)
            cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
            cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
            print(cluster_1, cluster_2)
            # update global learners by cluster 1
            learners = [client.learners_ensemble[min_sim_cluster_index] for i, (client, index) in enumerate(min_clients_by_clusters) if i in cluster_1]
            
            sampled_weights = torch.tensor([client.n_train_samples for i, (client, index) in enumerate(min_clients_by_clusters) if i in cluster_1])
            average_learners(learners, self.global_learners_ensemble[min_sim_cluster_index], weights=sampled_weights / sum(sampled_weights))
            # add new clusters
            for client in self.clients:
                client.add_learner(min_sim_cluster_index)
            for client in self.test_clients:
                client.add_learner(min_sim_cluster_index)
            self.add_learner(min_sim_cluster_index)
            # change the parameters of added clusters
            learners = [client.learners_ensemble[min_sim_cluster_index] for i, (client, index) in enumerate(min_clients_by_clusters) if i in cluster_2]
            
            sampled_weights = torch.tensor([client.n_train_samples for i, (client, index) in enumerate(min_clients_by_clusters) if i in cluster_2])
            average_learners(learners, self.global_learners_ensemble[-1], weights=sampled_weights / sum(sampled_weights))

            # hard clustering after splitting the clusters
            # for i in cluster_1:
            #     client = min_clients_by_clusters[i][0]
            #     client.learners_ensemble.learners_weights *= 0.0
            #     client.sample_learner_weights *= 0.0
            #     client.samples_weights *= 0.0
            #     client.learners_ensemble.learners_weights[min_sim_cluster_index] += 1.0
            #     client.sample_learner_weights[min_sim_cluster_index] += 1.0
            #     client.samples_weights[min_sim_cluster_index] += 1.0
                

            # for i in cluster_2:     
            #     client = min_clients_by_clusters[i][0]

            #     client.learners_ensemble.learners_weights *= 0.0
            #     client.sample_learner_weights *= 0.0
            #     client.samples_weights *= 0.0
            #     client.learners_ensemble.learners_weights[-1] += 1.0
            #     client.sample_learner_weights[-1] += 1.0
            #     client.samples_weights[-1] += 1.0
        print(decide_removes)

        # remove clusters
        for i, cluster_index in enumerate(decide_removes):
            for client in self.clients:
                client.remove_learner(cluster_index - i)
            for client in self.test_clients:
                client.remove_learner(cluster_index - i)
            self.remove_learner(cluster_index - i)


        # update global base learners and discriminators
        if self.split:
            base_learners = [client.learners_ensemble.base_learner for client in self.sampled_clients]
            average_learners(base_learners, self.global_learners_ensemble.base_learner, weights=torch.ones((len(self.sampled_clients),)) / len(self.sampled_clients))

        if self.domain_disc:
            disc_learners = [client.learners_ensemble.domain_disc_learner for client in self.sampled_clients]
            average_learners(disc_learners, self.global_learners_ensemble.domain_disc_learner, weights=torch.ones((len(self.sampled_clients),)) / len(self.sampled_clients))

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()
        
        if self.c_round in [1, 20, 50, 100, 150, 200]:
            self.save_similarites(self.c_round)

    

        


class STOCFLAggregator(CentralizedAggregator):

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            tol=0.15,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False
    ):

        super(STOCFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )


        self.tol = tol

        # self.global_learners = [self.global_learners_ensemble]
        self.clusters_indices = [[i] for i in range(len(self.clients))]
        self.n_clusters = len(self.clusters_indices)
        self.cluster_learners = [deepcopy(self.global_learners_ensemble) for _ in range(self.n_clusters)]
        self.chosen_clients = set()
        self.chosen_client_representations = {}
        self.update_clients()


    def cos_sim(self, x, y):
        sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return sim


    def mix(self, diverse=False):
        
        sampled_indices = self.sample_clients()

        self.chosen_clients = self.chosen_clients | set(sampled_indices)


        clients_updates = np.zeros((self.n_clients_per_round, self.model_dim))


        for client_id, client in enumerate(self.sampled_clients):
            clients_updates[client_id] = client.step()

        for i, client_index in enumerate(sampled_indices):
            if client_index not in self.chosen_client_representations:
                self.chosen_client_representations[client_index] = clients_updates[i]
        
        cluster_representations = np.zeros((self.n_clusters, self.model_dim))
        for c, c_indices in enumerate(self.clusters_indices):
            for index in c_indices:
                c_indices_count = 0
                if index in self.chosen_client_representations:
                    cluster_representations[c] += self.chosen_client_representations[index]
                    c_indices_count += 1
                if c_indices_count > 0:
                    cluster_representations[c] = cluster_representations[c] / c_indices_count
        
        M = np.zeros((self.n_clusters, self.n_clusters))
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                M[i][j] = self.cos_sim(cluster_representations[i], cluster_representations[j])
        print(np.mean(M), np.max(M), np.min(M))
        # print(M)
        
        removed_clusters = set()
        merged_clusters = []
        for i in range(self.n_clusters):
            if i in removed_clusters:
                merged_clusters.append([])
                continue
            merged_clusters_i = []
            for j in range(i+1, self.n_clusters):
                if j in removed_clusters:
                    continue
                if M[i][j] > self.tol:
                    merged_clusters_i.append(j)
                    removed_clusters.add(j)
            merged_clusters.append(merged_clusters_i)
        print(removed_clusters)

        for i in range(self.n_clusters):
            if i in removed_clusters:
                continue
            for j in merged_clusters[i]:
                self.clusters_indices[i] = self.clusters_indices[i] + self.clusters_indices[j]
        
        self.clusters_indices = [self.clusters_indices[i] for i in range(self.n_clusters) if i not in removed_clusters]
        self.cluster_learners = [self.cluster_learners[i] for i in range(self.n_clusters) if i not in removed_clusters]
        self.n_clusters = len(self.clusters_indices)
        print(self.n_clusters)
        print(self.clusters_indices)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]
            
            sampled_weights = torch.tensor([client.n_train_samples for client in self.sampled_clients])
            average_learners(learners, learner, weights=sampled_weights / sum(sampled_weights))
        
        for cluster_id, cluster_learner_ensemble in enumerate(self.cluster_learners):
            for learner_id, learner in enumerate(cluster_learner_ensemble):
                learners = [client.learners_ensemble[learner_id] for i, client in enumerate(self.clients) if i in self.clusters_indices[cluster_id]]
                sampled_weights = torch.tensor([client.n_train_samples for i, client in enumerate(self.clients) if i in self.clusters_indices[cluster_id]])
                average_learners(learners, learner, weights=sampled_weights / sum(sampled_weights))  
        
        self.update_clients()
        
        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()



    def update_clients(self):
        for cluster_id, indices in enumerate(self.clusters_indices):
            current_cluster_learners = self.cluster_learners[cluster_id]

            for i in indices:
                if self.clients[i].learners_ensemble[0].phi_model is not None:
                    copy_model(
                        target=self.clients[i].learners_ensemble[0].phi_model,
                        source=self.global_learners_ensemble[0].phi_model
                    )
                for learner_id, learner in enumerate(self.clients[i].learners_ensemble):
                    copy_model(
                        target=learner.model,
                        source=current_cluster_learners[learner_id].model
                    )
                    if callable(getattr(learner.optimizer, "set_initial_params", None)):
                        learner.optimizer.set_initial_params(
                            self.global_learners_ensemble[learner_id].model.parameters()
                        )

        if self.c_round >= 1:
            for client in self.test_clients:
                losses = [self.cluster_learners[i].gather_losses(client.val_iterator).mean() for i in range(len(self.cluster_learners))]
                cluster_id = losses.index(min(losses))
                cluster_learners = self.cluster_learners[cluster_id]
                for learner_id, learner in enumerate(client.learners_ensemble):
                    copy_model(target=learner.model, source=cluster_learners[learner_id].model)            

        
        



        

        
        

        






class ICFLAggregator(CentralizedAggregator):
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            epsilon_1=4.0,
            alpha_0=0.85,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False
    ):

        super(ICFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )

        assert self.n_learners == 1, "ClusteredAggregator only supports single learner clients."

        self.epsilon_1 = epsilon_1
        self.alpha_0 = alpha_0

        self.global_learners = [self.global_learners_ensemble]
        self.clusters_indices = [[i for i in range(len(self.clients))]]
        self.chosen_clients_parameters = {}
        self.chosen_clients = set()
        self.n_clusters = 1

    def cos_sim(self, x, y):
        sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return torch.tensor(sim)
    
    def find_c_true(self, client_parameter):
        c_index = 0
        max = -1.5
        for i in range(1, self.n_clusters):
            sim = self.cos_sim(client_parameter, self.global_learners[i].learner[0].get_param_tensor().cpu().numpy())
            if sim >= max:
                c_index = i
        return c_index


    def mix(self, diverse=False):

        new_cluster_indices = []
        sampled_indices_total = []
        for cluster_id in range(self.n_clusters):
             
            # sample clients in this cluster
            sampled_indices = self.sample_clients_group(self.clusters_indices[cluster_id])
            sampled_indices_total += sampled_indices
            n_i = len(sampled_indices)
            # update clients and save their corresponding parameters
            for client_id in sampled_indices:
                self.chosen_clients_parameters[client_id] = self.clients[client_id].step().squeeze()
                self.chosen_clients.add(client_id)

            # incremental clustering (seems typo in original paper here)
            if cluster_id == 0 and self.n_clusters > 1:
                for client_id in sampled_indices:
                    c_index = self.find_c_true(self.chosen_clients_parameters[client_id])
                    self.clusters_indices[c_index].append(client_id)
                self.clusters_indices[0] = [client_id for client_id in self.clusters_indices[0] if client_id not in sampled_indices]

            # if K > 0, do not split S0
            if self.n_clusters > 1 and cluster_id == 0:
                new_cluster_indices.append(self.clusters_indices[cluster_id])
                continue
            # clustering
            if cluster_id == 0:
                cluster_clients = [client_id for client_id in self.clusters_indices[cluster_id] if client_id in sampled_indices]
            else:
                cluster_clients = [client_id for client_id in self.clusters_indices[cluster_id]]
            cluster_size = len(cluster_clients)
            A = torch.zeros(cluster_size, cluster_size)
            # print(A, cluster_clients)
            for i in range(cluster_size):
                for j in range(cluster_size):
                    A[i][j] = self.cos_sim(self.chosen_clients_parameters[cluster_clients[i]], self.chosen_clients_parameters[cluster_clients[j]])
        
            # print(A)
            _, lam, q = torch.svd(A)
            # print(lam.shape, q.shape)
            epsilon_2 = torch.log(torch.tensor(1 - self.alpha_0)) / (1 - torch.log(torch.tensor(self.epsilon_1)))
            if len(lam) >= 2:
                lam = lam[:2]
                q = q[:2]
                alpha_G = lam[0] / cluster_size
            else:
                alpha_G = 1.0
            alpha_t = 1 - (torch.exp(torch.tensor(1.0)) / self.epsilon_1) ** (self.c_round/epsilon_2)

            # print(alpha_G, alpha_t)
            if (alpha_G < alpha_t) and len(lam) >= 2:
                new_q = torch.zeros(2, cluster_size)
                new_q[0] = q[0]
                new_q[1] = q[1]
                new_q = new_q.T
                y_pred = KMeans(n_clusters=2).fit_predict(new_q)
                C_1_clients = [cluster_clients[i] for i in range(cluster_size) if y_pred[i] == 0]
                C_2_clients = [cluster_clients[i] for i in range(cluster_size) if y_pred[i] == 1]
                
                if cluster_id == 0:
                    C_0_clients = [client_id for client_id in self.clusters_indices[cluster_id] if client_id not in sampled_indices]
                    new_cluster_indices.append(C_0_clients)

                new_cluster_indices.append(C_1_clients)
                new_cluster_indices.append(C_2_clients)
            else:
                new_cluster_indices.append(cluster_clients)
        
        self.clusters_indices = new_cluster_indices
        
        self.n_clusters = len(self.clusters_indices)

        print(self.n_clusters)
        print(self.clusters_indices)

        self.global_learners = [deepcopy(self.clients[self.clusters_indices[i][0]].learners_ensemble) for i in range(1, self.n_clusters)]
        self.global_learners = [self.global_learners_ensemble] + self.global_learners
        # print(len(self.global_learners))

        clusters_samples = []
        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_clients = [self.clients[i] for i in indices]
            sampled_weights = torch.tensor([client.n_train_samples for i, client in enumerate(cluster_clients)]) 
            if cluster_id == 0:
                cluster_clients = [self.clients[i] for i in indices if i in sampled_indices_total]
                sampled_weights = torch.tensor([client.n_train_samples for i, client in enumerate(cluster_clients)])
                for samples in sampled_weights:
                    clusters_samples.append(samples)
                continue
            else:
                clusters_samples.append(torch.sum(sampled_weights))
                for learner_id in range(self.n_learners):
                    average_learners(
                        learners=[client.learners_ensemble[learner_id] for client in cluster_clients],
                        target_learner=self.global_learners[cluster_id][learner_id],
                        weights=sampled_weights / torch.sum(sampled_weights)
                    )
        
        cluster_0_learners = [self.clients[i].learners_ensemble for i in self.clusters_indices[0] if i in sampled_indices_total]
        for k in range(1, self.n_clusters):
            cluster_0_learners.append(self.global_learners[k])
        for learner_id in range(self.n_learners):
            average_learners(
                learners=[learners_ensemble[learner_id] for learners_ensemble in cluster_0_learners],
                target_learner=self.global_learners[0][learner_id],
                weights=torch.tensor(clusters_samples) / sum(clusters_samples)
            )

        print(sampled_indices_total)
        for cluster_id, indices in enumerate(self.clusters_indices):
            sampled_indices = [index for index in indices if index in sampled_indices_total]
            self.update_clients(cluster_id, sampled_indices)

        self.update_test_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_test_clients(self):
        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_learners = self.global_learners[cluster_id]

            for i in indices:
                self.clients[i].global_learners_ensemble = cluster_learners

        for client in self.test_clients:
            losses = [self.global_learners[i].gather_losses(client.val_iterator).mean() for i in range(len(self.global_learners))]
            cluster_id = losses.index(min(losses))
            cluster_learners = self.global_learners[cluster_id]
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=cluster_learners[learner_id].model)

    def update_clients(self, cluster_id, indices):
        # for cluster_id, indices in enumerate(self.clusters_indices):
        cluster_learners = self.global_learners[cluster_id]

        for i in indices:
            for learner_id, learner in enumerate(self.clients[i].learners_ensemble):
                copy_model(
                    target=learner.model,
                    source=cluster_learners[learner_id].model
                )

    def write_logs(self):
        # self.update_test_clients()
        test_train_acces = []
        test_test_acces = []

        for global_logger, clients, client_type in [
            (self.global_train_logger, self.clients, 'train'),
            (self.global_test_logger, self.test_clients, 'test')
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if client_type == 'test':
                    test_train_acces.append(train_acc)
                    test_test_acces.append(test_acc)

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")

                    with np.printoptions(precision=3, suppress=True):
                        print("Pi: ", client.learners_weights.numpy())

                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)
                with open('./logs/{}/results-{}-{}.txt'.format(self.experiment, self.method, self.suffix), 'a+') as f:
                    f.write('{}, {}, {}, {}\n'.format(global_train_loss, global_train_acc, global_test_loss, global_test_acc))

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        print("+" * 30)
        
        print('test train accs: ' + str(test_train_acces))
        print('test_test_acces: ' + str(test_test_acces))
        print('test train variance: ' + str(torch.std(torch.tensor(test_train_acces) + 0.0).item()))
        print('test test variance: ' + str(torch.std(torch.tensor(test_test_acces) + 0.0).item()))
        print('test train mean: ' + str(torch.mean(torch.tensor(test_train_acces) + 0.0).item()))
        print('test test mean: ' + str(torch.mean(torch.tensor(test_test_acces) + 0.0).item()))
        with open('./logs/{}/test-results-{}-{}.txt'.format(self.experiment, self.method, self.suffix), 'a+') as f:
            f.write('test train accs: ' + str(test_train_acces) + '\n')
            f.write('test_test_acces: ' + str(test_test_acces) + '\n')
            f.write('test train variance: ' + str(torch.std(torch.tensor(test_train_acces) + 0.0).item()) + '\n')
            f.write('test test variance: ' + str(torch.std(torch.tensor(test_test_acces) + 0.0).item()) + '\n')
            f.write('test train mean: ' + str(torch.mean(torch.tensor(test_train_acces) + 0.0).item()) + '\n')
            f.write('test test mean: ' + str(torch.mean(torch.tensor(test_test_acces) + 0.0).item()) + '\n')

        if self.verbose > 0:
            print("#" * 80)
        






    
            


class ClusteredAggregator(Aggregator):
    """
    Implements
     `Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints`.

     Follows implementation from https://github.com/felisat/clustered-federated-learning
    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            tol_1=0.4,
            tol_2=0.6,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False
    ):

        super(ClusteredAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )

        assert self.n_learners == 1, "ClusteredAggregator only supports single learner clients."
        assert self.sampling_rate == 1.0, f"`sampling_rate` is {sampling_rate}, should be {1.0}," \
                                          f" ClusteredAggregator only supports full clients participation."

        self.tol_1 = tol_1
        self.tol_2 = tol_2

        self.global_learners = [self.global_learners_ensemble]
        self.clusters_indices = [np.arange(len(clients)).astype("int")]
        self.n_clusters = 1

    def mix(self, diverse=False):
        clients_updates = np.zeros((self.n_clients, self.n_learners, self.model_dim))

        for client_id, client in enumerate(self.clients):
            clients_updates[client_id] = client.step()

        max_N = 100
        if len(self.clusters_indices) < max_N:

            similarities = np.zeros((self.n_learners, self.n_clients, self.n_clients))

            for learner_id in range(self.n_learners):
                similarities[learner_id] = pairwise_distances(clients_updates[:, learner_id, :], metric="cosine")

            similarities = similarities.mean(axis=0)

            new_cluster_indices = []
            max_update_norm, mean_update_norm = 0, 0
            for indices in self.clusters_indices:
                max_update_norm = np.zeros(self.n_learners)
                mean_update_norm = np.zeros(self.n_learners)

                for learner_id in range(self.n_learners):
                    max_update_norm[learner_id] = LA.norm(clients_updates[indices], axis=1).max()
                    mean_update_norm[learner_id] = LA.norm(np.mean(clients_updates[indices], axis=0))

                max_update_norm = max_update_norm.mean()
                mean_update_norm = mean_update_norm.mean()

                if mean_update_norm < self.tol_1 and max_update_norm > self.tol_2 and len(indices) > 2:
                    clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete")
                    clustering.fit(similarities[indices][:, indices])
                    cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
                    cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
                    new_cluster_indices += [cluster_1, cluster_2]
                else:
                    new_cluster_indices += [indices]
            
            # print(new_cluster_indices)

            # if len(new_cluster_indices) <= max_N:
            self.clusters_indices = new_cluster_indices

        self.n_clusters = len(self.clusters_indices)

        print(self.n_clusters, max_update_norm, mean_update_norm)

        self.global_learners = [deepcopy(self.clients[self.clusters_indices[i][0]].learners_ensemble) for i in range(self.n_clusters)]

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_clients = [self.clients[i] for i in indices]
            for learner_id in range(self.n_learners):
                average_learners(
                    learners=[client.learners_ensemble[learner_id] for client in cluster_clients],
                    target_learner=self.global_learners[cluster_id][learner_id],
                    weights=self.clients_weights[indices] / self.clients_weights[indices].sum()
                )

        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_learners = self.global_learners[cluster_id]

            for i in indices:
                for learner_id, learner in enumerate(self.clients[i].learners_ensemble):
                    copy_model(
                        target=learner.model,
                        source=cluster_learners[learner_id].model
                    )

        for client in self.test_clients:
            losses = [self.global_learners[i].gather_losses(client.val_iterator).mean() for i in range(len(self.global_learners))]
            cluster_id = losses.index(min(losses))
            cluster_learners = self.global_learners[cluster_id]
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=cluster_learners[learner_id].model)

    def update_test_clients(self):
        pass


class FedSoftAggregator(Aggregator):

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            alpha=0.5,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False
    ):
        super(FedSoftAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )

        self.global_learners = [self.global_learners_ensemble, deepcopy(self.global_learners_ensemble), deepcopy(self.global_learners_ensemble)]
        self.global_learners[1][0].model.classifier[1] = nn.Linear(self.global_learners[1][0].model.classifier[1].in_features, self.clients[0].class_number)
        # self.global_learners[1][0].model.fc = nn.Linear(self.global_learners[1][0].model.fc.in_features, self.clients[0].class_number)
        # self.global_learners[1][0].model.output = nn.Linear(self.global_learners[1][0].model.output.in_features, self.clients[0].class_number)
        self.global_learners[2][0].model.classifier[1] = nn.Linear(self.global_learners[2][0].model.classifier[1].in_features, self.clients[0].class_number)
        # self.global_learners[2][0].model.fc = nn.Linear(self.global_learners[2][0].model.fc.in_features, self.clients[0].class_number)
        # self.global_learners[2][0].model.output = nn.Linear(self.global_learners[2][0].model.output.in_features, self.clients[0].class_number)
        self.n_clusters = 3
        self.clusters_weights = torch.ones(len(clients), self.n_clusters) / self.n_clusters
        self.clients_weights = torch.ones(len(clients), self.n_clusters) / len(clients)
        self.sigma = 1e-4

    def mix(self, diverse=False):

        self.sample_clients()

        for k, client in enumerate(self.clients):
            client_losses = torch.zeros(self.n_clusters, client.n_train_samples)
            for i in range(len(self.global_learners)):
                losses = self.global_learners[i].gather_losses(client.val_iterator)[0]
                client_losses[i] = losses
            # print(client_losses)
            max_values, max_indices = torch.max(client_losses, dim=0)
            client_cluster_weights = torch.zeros(self.n_clusters, client.n_train_samples)
            for i in range(self.n_clusters):
                for j in range(client.n_train_samples):
                    client_cluster_weights[i][j] = 1 if max_indices[j] == i else 0
            client_cluster_weights = torch.sum(client_cluster_weights, dim=1)
            client_cluster_weights[client_cluster_weights < 1e-4] = 1e-4
            # client_cluster_weights = torch.max(client_cluster_weights, dim=self.sigma)
            # client_cluster_weights = client_cluster_weights / torch.sum(client_cluster_weights)
            self.clusters_weights[k] = client_cluster_weights
        
        self.clients_weights = self.clusters_weights / torch.sum(self.clusters_weights, dim=0)
        self.clusters_weights = (self.clusters_weights.T / torch.sum(self.clusters_weights, dim=1)).T

        # print(self.clients_weights)
        # print(self.clusters_weights)

        # print(self.clients_weights.T[0], sum(self.clients_weights.T[0]))

        for client in self.sampled_clients:
            client.step()

        # average_learners(
        #             learners=[client.learners_ensemble[learner_id] for client in cluster_clients],
        #             target_learner=self.global_learners[cluster_id][learner_id],
        #             weights=self.clients_weights[indices] / self.clients_weights[indices].sum()
        #         )

        for cluster_id in range(self.n_clusters):
            for learner_id in range(self.n_learners):
                average_learners(
                        learners=[client.learners_ensemble[1] for client in self.sampled_clients],
                        target_learner=self.global_learners[cluster_id][learner_id],
                        weights=self.clients_weights.T[cluster_id]
                    )
        
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client_id, client in enumerate(self.clients):
            average_learners(
                learners=[learner[0] for learner in self.global_learners],
                target_learner=client.learners_ensemble[0],
                weights=self.clusters_weights[client_id]
            )

        for client in self.test_clients:
            losses = [self.global_learners[i].gather_losses(client.val_iterator).mean() for i in range(len(self.global_learners))]
            cluster_id = losses.index(min(losses))
            cluster_learners = self.global_learners[cluster_id]
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=cluster_learners[0].model)
        

        

        
        


        
            





class AgnosticAggregator(CentralizedAggregator):
    """
    Implements
     `Agnostic Federated Learning`__(https://arxiv.org/pdf/1902.00146.pdf).

    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            lr_lambda,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False
    ):
        super(AgnosticAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )

        self.lr_lambda = lr_lambda

    def mix(self):
        self.sample_clients()

        clients_losses = []
        for client in self.sampled_clients:
            client_losses = client.step()
            clients_losses.append(client_losses)

        clients_losses = torch.tensor(clients_losses)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]

            average_learners(
                learners=learners,
                target_learner=learner,
                weights=self.clients_weights,
                average_gradients=True
            )

        # update parameters
        self.global_learners_ensemble.optimizer_step()

        # update clients weights
        self.clients_weights += self.lr_lambda * clients_losses.mean(dim=1)
        self.clients_weights = simplex_projection(self.clients_weights)

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class FFLAggregator(CentralizedAggregator):
    """
    Implements q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            lr,
            q=1,
            sampling_rate=1.,
            sample_with_replacement=True,
            test_clients=None,
            verbose=0,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,
            domain_disc=False
    ):
        super(FFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )

        self.q = q
        self.lr = lr
        assert self.sample_with_replacement, 'FFLAggregator only support sample with replacement'

    def mix(self):
        self.sample_clients()

        hs = 0
        for client in self.sampled_clients:
            hs += client.step(lr=self.lr)

        hs /= (self.lr * len(self.sampled_clients))  # take account for the lr used inside optimizer

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]
            average_learners(
                learners=learners,
                target_learner=learner,
                weights=hs*torch.ones(len(learners)),
                average_params=False,
                average_gradients=True
            )

        # update parameters
        self.global_learners_ensemble.optimizer_step()

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class DecentralizedAggregator(Aggregator):
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            mixing_matrix,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=True,
            test_clients=None,
            verbose=0,
            seed=None,
            experiment=None,
            method=None,
            suffix=None,
            split=False,domain_disc=False):

        super(DecentralizedAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )

        self.mixing_matrix = mixing_matrix
        assert self.sampling_rate >= 1, "partial sampling is not supported with DecentralizedAggregator"

    def update_clients(self):
        pass

    def mix(self):
        # update local models
        for client in self.clients:
            client.step()

        # mix models
        mixing_matrix = torch.tensor(
            self.mixing_matrix.copy(),
            dtype=torch.float32,
            device=self.device
        )

        for learner_id, global_learner in enumerate(self.global_learners_ensemble):
            state_dicts = [client.learners_ensemble[learner_id].model.state_dict() for client in self.clients]

            for key, param in global_learner.model.state_dict().items():
                shape_ = param.shape
                models_params = torch.zeros(self.n_clients, int(np.prod(shape_)), device=self.device)

                for ii, sd in enumerate(state_dicts):
                    models_params[ii] = sd[key].view(1, -1)

                models_params = mixing_matrix @ models_params

                for ii, sd in enumerate(state_dicts):
                    sd[key] = models_params[ii].view(shape_)

            for client_id, client in enumerate(self.clients):
                client.learners_ensemble[learner_id].model.load_state_dict(state_dicts[client_id])

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()