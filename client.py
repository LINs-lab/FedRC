import torch.nn.functional as F

from copy import deepcopy
from utils.torch_utils import *
from math import log



class Client(object):
    r"""Implements one clients

    Attributes
    ----------
    learners_ensemble
    n_learners

    train_iterator

    val_iterator

    test_iterator

    train_loader

    n_train_samples

    n_test_samples

    samples_weights

    local_steps

    logger

    tune_locally:

    Methods
    ----------
    __init__
    step
    write_logs
    update_sample_weights
    update_learners_weights

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            data_type = 0,
            feature_types = None,
            class_number = 10
    ):

        self.learners_ensemble = learners_ensemble
        self.n_learners = len(self.learners_ensemble)
        self.tune_locally = tune_locally
        self.data_type = data_type
        self.feature_types = feature_types
        self.cluster = torch.ones(self.n_learners) / self.n_learners
        self.class_number = class_number
        self.global_learners_ensemble = None

        if self.tune_locally:
            self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)
        else:
            self.tuned_learners_ensemble = None

        self.binary_classification_flag = self.learners_ensemble.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.samples_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners
        self.samples_weights_momentum = torch.zeros(self.n_learners, self.n_train_samples) / self.n_learners
        self.samples_weights_momentum_1 = torch.zeros(self.n_learners, self.n_train_samples) / self.n_learners
        self.samples_flags = self.samples_weights
        self.global_flags_mean = torch.ones((self.n_learners,)) / self.n_learners
        self.global_flags_std = torch.ones((self.n_learners,)) / self.n_learners

        self.distances = torch.zeros((self.n_learners,))

        self.local_steps = local_steps

        self.counter = 0
        self.logger = logger

        self.mean_I = 0.0

        # self.labels_weights = torch.ones(self.n_learners, self.n_train_samples * 80) / self.n_learners
        self.labels_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners
        self.labels_mask = torch.zeros(class_number, self.n_train_samples)
        self.label_stats = self.get_label_stats()
        self.need_new_model = False

        self.labels_learner_weights = torch.ones(self.n_learners, self.class_number) / self.class_number

        self.label_learners_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners

        self.sample_learner_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners

        self.entropy = torch.ones(self.n_train_samples)


    def update_labels_weights(self, labels_weights):
        for i, y in enumerate(self.train_iterator.dataset.targets):
            if self.labels_weights.shape[1] > self.n_train_samples:
                for y_i_index, y_i in enumerate(y):
                    for j in range(self.n_learners):
                        self.labels_weights[j][i * 80 + y_i_index] = labels_weights[j][y_i]
            else:
                for j in range(self.n_learners):
                    self.labels_weights[j][i] = labels_weights[j][y]
        for i, learner in enumerate(self.learners_ensemble.learners):
            learner.labels_weights = labels_weights[i]
    def get_label_stats(self):
        labels = {}
        for i, y in enumerate(self.train_iterator.dataset.targets):
            if self.labels_weights.shape[1] > self.n_train_samples:
                for y_i in y:
                    if y_i in labels:
                        labels[y_i] += 1
                    else:
                        labels[y_i] = 1
                    self.labels_mask[y_i][i] = 1.0
            else:
                if y in labels:
                    labels[y] += 1
                else:
                    labels[y] = 1
                self.labels_mask[y][i] = 1.0
        return labels

    def get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch

    def add_learner(self, index):
        # new_learner = deepcopy(self.learners_ensemble.learners[index])
        self.n_learners += 1
        self.learners_ensemble.add_learner(index)
        
        
        # self.samples_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners
        self.labels_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners
        # self.sample_learner_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners


        self.samples_weights = torch.cat((self.samples_weights, self.samples_weights[index].unsqueeze(0) / 2), 0)
        self.samples_weights[index] = self.samples_weights[index] / 2
        self.sample_learner_weights = torch.cat((self.sample_learner_weights, self.sample_learner_weights[index].unsqueeze(0) / 2), 0)
        self.sample_learner_weights[index] = self.sample_learner_weights[index] / 2

        self.samples_weights = self.samples_weights / torch.sum(self.samples_weights, dim=0)
        self.sample_learner_weights = self.sample_learner_weights / torch.sum(self.sample_learner_weights, dim=0)

        self.distances = torch.cat((self.distances, self.distances[index].unsqueeze(0)), 0)

        self.update_learner_labels_weights()

    def remove_learner(self, learner_index):
        self.n_learners -= 1
        self.learners_ensemble.remove_learner(learner_index)
        self.samples_weights = torch.cat((self.samples_weights[:learner_index], self.samples_weights[learner_index+1:]), 0)
        self.labels_weights = torch.cat((self.labels_weights[:learner_index], self.labels_weights[learner_index+1:]), 0)
        self.samples_weights = self.samples_weights / torch.sum(self.samples_weights, dim=0)
        self.samples_weights_momentum = torch.cat((self.samples_weights_momentum[:learner_index], self.samples_weights_momentum[learner_index+1:]), 0)
        self.samples_weights_momentum_1 = torch.cat((self.samples_weights_momentum_1[:learner_index], self.samples_weights_momentum_1[learner_index+1:]), 0)

        self.sample_learner_weights = torch.cat((self.sample_learner_weights[:learner_index], self.sample_learner_weights[learner_index+1:]), 0)
        self.sample_learner_weights = self.sample_learner_weights / torch.sum(self.sample_learner_weights, dim=0)

        self.distances = torch.cat((self.distances[:learner_index], self.distances[learner_index + 1:]), 0)

    def step_line_search(self, new_params, initial_params):
        alpha = 2.0 ** 3
        #get baseline L
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        baseline_L = torch.sum(torch.exp(- all_losses - torch.log(self.labels_weights)) * self.samples_weights) / self.n_train_samples
        # get updates
        initial_state_dicts = [deepcopy(learner.model.state_dict()) for learner in initial_params.learners]
        params_updates_dicts = [deepcopy(learner.model.state_dict()) for learner in initial_params.learners]
        for i in range(self.n_learners):
            new_state_dicts = new_params.learners[i].model.state_dict()
            for k, v in initial_state_dicts[i].items():
                params_updates_dicts[i][k] = new_state_dicts[k].clone().detach() - initial_state_dicts[i][k].clone().detach()
        
        # search
        for i in range(6):
            new_search_params = [deepcopy(learner.model.state_dict()) for learner in initial_params.learners]
            for i in range(self.n_learners):
                for k, v in params_updates_dicts[i].items():
                    new_search_params[i][k] = new_search_params[i][k] + alpha * params_updates_dicts[i][k]
                self.learners_ensemble.learners[i].model.load_state_dict(new_search_params[i])

            all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
            L = torch.sum(torch.exp(- all_losses - torch.log(self.labels_weights)) * self.samples_weights) / self.n_train_samples
            if L > baseline_L:
                return
            alpha = alpha * 0.5

        for i in range(self.n_learners):
            self.learners_ensemble.learners[i].model.load_state_dict(initial_state_dicts[i])
                





    def step(self, single_batch_flag=False, diverse=True, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1

        # initial_params = deepcopy(self.learners_ensemble)

        self.update_sample_weights()
        self.update_learners_weights()


        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights,
                    entropy=self.entropy
                )

        # self.step_line_search(self.learners_ensemble, initial_params)

        # TODO: add flag arguments to use `free_gradients`
        # self.learners_ensemble.free_gradients()

        return client_updates

    def write_logs(self):

        if self.tune_locally:
            self.update_tuned_learners()
            train_loss, train_acc = self.tuned_learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.tuned_learners_ensemble.evaluate_iterator(self.test_iterator)
        elif self.global_learners_ensemble is not None:
            # print('in global ensemble')
            train_loss, train_acc = self.global_learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.global_learners_ensemble.evaluate_iterator(self.test_iterator)
        else:
            train_loss, train_acc = self.learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.learners_ensemble.evaluate_iterator(self.test_iterator)
            # print(train_loss, train_acc, test_loss, test_acc)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc

    def update_sample_weights(self):
        pass

    def update_learners_weights(self):
        pass

    def update_tuned_learners(self):
        if not self.tune_locally:
            return
        self.tuned_learners_ensemble.learners_weights = deepcopy(self.learners_ensemble.learners_weights)
        for learner_id, learner in enumerate(self.tuned_learners_ensemble):
            copy_model(source=self.learners_ensemble[learner_id].model, target=learner.model)
            learner.fit_epochs(self.train_iterator, 1, weights=self.samples_weights[learner_id])


class MixtureClient(Client):

    def update_learner_labels_weights(self):
        self.labels_learner_weights = torch.zeros(self.n_learners, self.class_number) / self.class_number
        if self.labels_weights.shape[1] > self.n_train_samples:
            for i, y in enumerate(self.train_iterator.dataset.targets):
                for y_i in y:
                    for j in range(self.n_learners):
                        self.labels_learner_weights[j][y_i] += self.samples_weights[j][i]
        else:
            for i, y in enumerate(self.train_iterator.dataset.targets):
                for j in range(self.n_learners):
                    self.labels_learner_weights[j][y] += self.samples_weights[j][i]
                    
    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T


    def update_learners_weights(self):
        # print(self.learners_ensemble.learners_weights, end='  ')
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
        weights = self.learners_ensemble.learners_weights
        self.cluster = weights


class MixtureClient_SW(Client):

    def update_learner_labels_weights(self):
        self.labels_learner_weights = torch.zeros(self.n_learners, self.class_number) / self.class_number
        if self.labels_weights.shape[1] > self.n_train_samples:
            for i, y in enumerate(self.train_iterator.dataset.targets):
                for y_i in y:
                    for j in range(self.n_learners):
                        self.labels_learner_weights[j][y_i] += self.samples_weights[j][i]
        else:
            for i, y in enumerate(self.train_iterator.dataset.targets):
                for j in range(self.n_learners):
                    self.labels_learner_weights[j][y] += self.samples_weights[j][i]
                    
    # def update_sample_weights(self):
    #     all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
    #     self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T

    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        L = - all_losses.T

        self.mean_I = torch.exp(torch.log(self.sample_learner_weights.T) - all_losses.T).T
        self.mean_I = torch.mean(torch.sum(self.mean_I,dim=0))

        samples_weights_1 = F.softmax(torch.log(self.sample_learner_weights.T) + L, dim=1).T
        self.new_samples_weights = F.softmax(torch.log(self.learners_ensemble.learners_weights) + L, dim=1).T

        self.samples_weights = samples_weights_1

    def update_learners_weights(self):
        mu = 0.0
        self.learners_ensemble.learners_weights = self.new_samples_weights.mean(dim=1)
        weights = self.learners_ensemble.learners_weights
        # print(weights)
        self.cluster = weights
        self.sample_learner_weights = (mu * self.samples_weights.T + (1-mu) * self.learners_ensemble.learners_weights).T
        self.update_learner_labels_weights()



    # def update_learners_weights(self):
    #     # print(self.learners_ensemble.learners_weights, end='  ')
    #     self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
    #     weights = self.learners_ensemble.learners_weights
    #     self.cluster = weights

class FedRC(Client):

    def update_learner_labels_weights(self):
        self.labels_learner_weights = torch.zeros(self.n_learners, self.class_number) / self.class_number
        if self.labels_weights.shape[1] > self.n_train_samples:
            for i, y in enumerate(self.train_iterator.dataset.targets):
                for y_i in y:
                    for j in range(self.n_learners):
                        self.labels_learner_weights[j][y_i] += self.samples_weights[j][i]
        else:
            for i, y in enumerate(self.train_iterator.dataset.targets):
                for j in range(self.n_learners):
                    self.labels_learner_weights[j][y] += self.samples_weights[j][i]

    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        L = - all_losses.T - torch.log(self.labels_weights.T)
        # L = L.reshape(self.n_train_samples, 80, self.n_learners)
        # L = torch.sum(L, dim=1)
        # self.mean_I = torch.exp(torch.log(self.learners_ensemble.learners_weights) + L).T
        # self.mean_I = torch.mean(torch.sum(self.mean_I,dim=1))
        self.mean_I = torch.exp(torch.log(self.learners_ensemble.learners_weights) - all_losses.T).T
        self.mean_I = torch.mean(torch.sum(self.mean_I,dim=1))

        new_samples_weights = F.softmax(torch.log(self.learners_ensemble.learners_weights) + L, dim=1).T
        self.samples_weights = new_samples_weights


    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
        weights = self.learners_ensemble.learners_weights
        # self.learners_ensemble.learners_weights = (self.learners_ensemble.learners_weights == max(self.learners_ensemble.learners_weights)).float()
        self.cluster = weights
        self.update_learner_labels_weights()




class FedRC_SW(FedRC):

    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        L = - all_losses.T - torch.log(self.labels_weights.T)

        self.mean_I = torch.exp(torch.log(self.sample_learner_weights.T) - all_losses.T).T
        self.mean_I = torch.mean(torch.sum(self.mean_I,dim=0))

        samples_weights_1 = F.softmax(torch.log(self.sample_learner_weights.T) + L, dim=1).T
        self.new_samples_weights = F.softmax(torch.log(self.learners_ensemble.learners_weights) + L, dim=1).T

        self.samples_weights = samples_weights_1

    
    def update_learners_weights_single(self):
        mu = 0.4
        self.learners_ensemble.learners_weights = self.new_samples_weights.mean(dim=1)
        
        weights = self.learners_ensemble.learners_weights
        # print(weights)
        self.cluster = weights
        self.sample_learner_weights = (mu * self.samples_weights.T + (1-mu) * self.learners_ensemble.learners_weights).T
        self.update_learner_labels_weights()

        new_learners_weights = torch.zeros_like(self.learners_ensemble.learners_weights)
        new_learners_weights[self.learners_ensemble.learners_weights == torch.max(self.learners_ensemble.learners_weights)] = 1.0
        self.learners_ensemble.learners_weights = new_learners_weights


    def update_learners_weights(self):
        mu = 0.4
        self.learners_ensemble.learners_weights = self.new_samples_weights.mean(dim=1)
        weights = self.learners_ensemble.learners_weights
        # print(weights)
        self.cluster = weights
        self.sample_learner_weights = (mu * self.samples_weights.T + (1-mu) * self.learners_ensemble.learners_weights).T
        self.update_learner_labels_weights()



class FedRC_Adam(Client):

    def update_learner_labels_weights(self):
        self.labels_learner_weights = torch.ones(self.n_learners, self.class_number) / self.class_number
        for i, y in enumerate(self.train_iterator.dataset.targets):
            for j in range(self.n_learners):
                self.labels_learner_weights[j][y] += self.samples_weights[j][i]

    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        L = - all_losses.T - torch.log(self.labels_weights.T)

        new_samples_weights = F.softmax(torch.log(self.learners_ensemble.learners_weights) + L, dim=1).T
        # add adam
        alpha = 1.0
        beta_1 = 0.5
        beta_2 = 0.5
        # beta_1 = 0.9
        # beta_2 = 0.99
        epsilon = 1e-8
        g_t = self.samples_weights - new_samples_weights
        self.samples_weights_momentum = beta_1 * self.samples_weights_momentum + (1 - beta_1) * g_t
        self.samples_weights_momentum_1 = beta_2 * self.samples_weights_momentum_1 + (1 - beta_2) * (g_t ** 2)
        m_t_hat = self.samples_weights_momentum / (1 - beta_1)
        v_t_hat = self.samples_weights_momentum_1 / (1 - beta_2)
        self.samples_weights = self.samples_weights - alpha * m_t_hat / ((v_t_hat ** 0.5) + epsilon)
        # normalize
        self.samples_weights = torch.max(self.samples_weights, torch.zeros_like(self.samples_weights))
        self.samples_weights = self.samples_weights / torch.sum(self.samples_weights, dim=0)


    def update_learners_weights(self):
        # print(self.learners_ensemble.learners_weights, end='  ')
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
        weights = self.learners_ensemble.learners_weights
        self.cluster = weights
        self.update_learner_labels_weights()

class FedRC_DP(Client):
    def update_learner_labels_weights(self):
        self.labels_learner_weights = torch.zeros(self.n_learners, self.class_number) / self.class_number
        for i, y in enumerate(self.train_iterator.dataset.targets):
            for j in range(self.n_learners):
                self.labels_learner_weights[j][y] += self.samples_weights[j][i]
        self.labels_learner_weights += torch.normal(torch.zeros_like(self.labels_learner_weights), std=100.0)

    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        L = - all_losses.T - torch.log(self.labels_weights.T)

        new_samples_weights = F.softmax(torch.log(self.learners_ensemble.learners_weights) + L, dim=1).T
        self.samples_weights = new_samples_weights


    def update_learners_weights(self):
        # print(self.learners_ensemble.learners_weights, end='  ')
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
        weights = self.learners_ensemble.learners_weights
        self.cluster = weights
        self.update_learner_labels_weights()


class FeSEM(Client):

    def add_learner(self, index):
        # new_learner = deepcopy(self.learners_ensemble.learners[index])
        self.n_learners += 1
        self.learners_ensemble.add_learner(index)
        
        
        # self.samples_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners
        self.labels_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners
        # self.sample_learner_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners


        self.samples_weights = torch.cat((self.samples_weights, self.samples_weights[index].unsqueeze(0) / 2), 0)
        self.samples_weights[index] = self.samples_weights[index] / 2
        self.sample_learner_weights = torch.cat((self.sample_learner_weights, self.sample_learner_weights[index].unsqueeze(0) / 2), 0)
        self.sample_learner_weights[index] = self.sample_learner_weights[index] / 2

        self.samples_weights = self.samples_weights / torch.sum(self.samples_weights, dim=0)
        self.sample_learner_weights = self.sample_learner_weights / torch.sum(self.sample_learner_weights, dim=0)

        self.distances = torch.cat((self.distances, self.distances[index].unsqueeze(0)), 0)

        self.update_learner_labels_weights()
        self.update_sample_weights()
        self.update_learners_weights()

    def remove_learner(self, learner_index):
        self.n_learners -= 1
        self.learners_ensemble.remove_learner(learner_index)
        self.samples_weights = torch.cat((self.samples_weights[:learner_index], self.samples_weights[learner_index+1:]), 0)
        self.labels_weights = torch.cat((self.labels_weights[:learner_index], self.labels_weights[learner_index+1:]), 0)
        self.samples_weights = self.samples_weights / torch.sum(self.samples_weights, dim=0)
        self.samples_weights_momentum = torch.cat((self.samples_weights_momentum[:learner_index], self.samples_weights_momentum[learner_index+1:]), 0)
        self.samples_weights_momentum_1 = torch.cat((self.samples_weights_momentum_1[:learner_index], self.samples_weights_momentum_1[learner_index+1:]), 0)

        self.sample_learner_weights = torch.cat((self.sample_learner_weights[:learner_index], self.sample_learner_weights[learner_index+1:]), 0)
        self.sample_learner_weights = self.sample_learner_weights / torch.sum(self.sample_learner_weights, dim=0)

        self.distances = torch.cat((self.distances[:learner_index], self.distances[learner_index + 1:]), 0)
        self.update_sample_weights()
        self.update_learners_weights()

    def step(self, single_batch_flag=False, diverse=True, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1

        # initial_params = deepcopy(self.learners_ensemble)

        # self.update_sample_weights()
        # self.update_learners_weights()


        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights,
                    entropy=self.entropy
                )

        # self.step_line_search(self.learners_ensemble, initial_params)

        # TODO: add flag arguments to use `free_gradients`
        # self.learners_ensemble.free_gradients()

        return client_updates

    def update_learner_labels_weights(self):
        self.labels_learner_weights = torch.zeros(self.n_learners, self.class_number) / self.class_number
        if self.labels_weights.shape[1] > self.n_train_samples:
            for i, y in enumerate(self.train_iterator.dataset.targets):
                for y_i in y:
                    for j in range(self.n_learners):
                        self.labels_learner_weights[j][y_i] += self.samples_weights[j][i]
        else:
            for i, y in enumerate(self.train_iterator.dataset.targets):
                for j in range(self.n_learners):
                    self.labels_learner_weights[j][y] += self.samples_weights[j][i]

    def update_sample_weights(self):
        if sum(self.distances) == 0.0:
            all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
            mean_losses = torch.mean(all_losses, 1).squeeze()
            cluster_index = torch.nonzero(mean_losses == min(mean_losses))
        else:
            cluster_index = torch.nonzero(self.distances == min(self.distances))
        # print(self.distances, self.n_learners, cluster_index)
        self.samples_weights = torch.zeros(self.n_learners, self.n_train_samples)
        self.samples_weights[cluster_index[0],:] = 1.0



    def update_learners_weights(self):
        # print(self.learners_ensemble.learners_weights, end='  ')
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
        weights = self.learners_ensemble.learners_weights
        self.cluster = weights

class FedSoft(Client):

    def update_sample_weights(self):
        self.samples_weights = torch.zeros(self.n_learners, self.n_train_samples)
        self.samples_weights[1,:] = 1.0

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
        weights = self.learners_ensemble.learners_weights

        # if callable(getattr(self.learners_ensemble[1].optimizer, "set_initial_params", None)):
        self.learners_ensemble[1].optimizer.set_initial_params([param for param in self.learners_ensemble[0].model.parameters() if param.requires_grad])

    


class IFCA(Client):
    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        mean_losses = torch.mean(all_losses, 1).squeeze()
        cluster_index = torch.nonzero(mean_losses == min(mean_losses))
        self.samples_weights = torch.zeros(self.n_learners, self.n_train_samples)
        self.samples_weights[cluster_index[0],:] = 1.0

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
        weights = self.learners_ensemble.learners_weights
        self.cluster = weights



class MixtureClientAdapt(MixtureClient):
    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        self.samples_weights_1 = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T
        self.samples_weights_2 = F.softmax((torch.log(self.learners_ensemble.learners_weights) + torch.log(self.labels_weights)), dim=1).T
        self.samples_weights = self.samples_weights_1 - self.samples_weights_2


class AgnosticFLClient(Client):
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            data_type=0,
            feature_types = None
    ):
        super(AgnosticFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."

    def step(self, *args, **kwargs):
        self.counter += 1

        batch = self.get_next_batch()
        losses = self.learners_ensemble.compute_gradients_and_loss(batch)

        return losses


class FFLClient(Client):
    r"""
    Implements client for q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            q=1,
            tune_locally=False,
            data_type=0,
            feature_types = None
    ):
        super(FFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."
        self.q = q

    def step(self, lr, *args, **kwargs):

        hs = 0
        for learner in self.learners_ensemble:
            initial_state_dict = self.learners_ensemble[0].model.state_dict()
            learner.fit_epochs(iterator=self.train_iterator, n_epochs=self.local_steps)

            client_loss, _ = learner.evaluate_iterator(self.train_iterator)
            client_loss = torch.tensor(client_loss)
            client_loss += 1e-10

            # assign the difference to param.grad for each param in learner.parameters()
            differentiate_learner(
                target=learner,
                reference_state_dict=initial_state_dict,
                coeff=torch.pow(client_loss, self.q) / lr
            )

            hs = self.q * torch.pow(client_loss, self.q-1) * torch.pow(torch.linalg.norm(learner.get_grad_tensor()), 2)
            hs /= torch.pow(torch.pow(client_loss, self.q), 2)
            hs += torch.pow(client_loss, self.q) / lr

        return hs / len(self.learners_ensemble)
    
class ACGMixtureClient(Client):
    def __init__(self, learners_ensemble, train_iterator, val_iterator, test_iterator, logger, local_steps, save_path,
                 tune_locally=False):
        super().__init__(learners_ensemble, train_iterator, val_iterator, test_iterator, logger, local_steps, save_path,
                         tune_locally)
        self.learners_ensemble.initialize_gmm(iterator=train_iterator)

    def update_sample_weights(self):
        self.samples_weights = self.learners_ensemble.calc_samples_weights(self.val_iterator)

    def update_learners_weights(self):  # calculate pi, mu and Var
        self.learners_ensemble.m_step(self.samples_weights, self.val_iterator)
    """
    " Only update gmm
    """
    def step(self, single_batch_flag=False, n_iter=1, *args, **kwargs):
        self.counter += 1

        # self.learners_ensemble.initialize_gmm(iterator=self.train_iterator)
        """
        " EM step
        """
        for _ in range(n_iter):
            self.update_sample_weights()  # update q(x)
            self.update_learners_weights()  # update pi, mu and Var

        sum_samples_weights = self.samples_weights.sum(dim=1)
        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=sum_samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=sum_samples_weights
                )

        self.learners_ensemble.free_gradients()
        # self.clear_models()

        return client_updates

    def unseen_step(self, single_batch_flag=False, n_iter=1, *args, **kwargs):
        self.counter += 1

        """
        " EM step
        """
        for _ in range(n_iter):
            self.update_sample_weights()  # update q(x)

    def gmm_step(self, single_batch_flag=False, n_iter=1, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        # self.reload_models()
        self.counter += 1

        # self.learners_ensemble.initialize_gmm(iterator=self.train_iterator)
        """
        " EM step
        """
        for _ in range(n_iter):
            self.update_sample_weights()  # update q(x)
            self.update_learners_weights()  # update pi, mu and Var

        # sum_samples_weights = self.samples_weights.sum(dim=1)
        # if single_batch_flag:
        #     batch = self.get_next_batch()
        #     client_updates = \
        #         self.learners_ensemble.fit_batch(
        #             batch=batch,
        #             weights=sum_samples_weights
        #         )
        # else:
        #     client_updates = \
        #         self.learners_ensemble.fit_epochs(
        #             iterator=self.train_iterator,
        #             n_epochs=self.local_steps,
        #             weights=sum_samples_weights
        #         )

        # self.learners_ensemble.free_gradients()
        # self.clear_models()

        # return client_updates
        return

    def ac_step(self):
        self.learners_ensemble.freeze_classifier()

        ac_client_update = \
            self.learners_ensemble.fit_ac_epochs(
                iterator=self.train_iterator,
                n_epochs=self.local_steps
            )

        self.learners_ensemble.unfreeze_classifier()

        return ac_client_update

    # def write_logs(self):
    #     if self.tune_locally:
    #         self.update_tuned_learners()

    #     if self.tune_locally:
    #         train_loss, train_acc = self.tuned_learners_ensemble.evaluate_iterator(self.val_iterator)
    #         test_loss, test_acc = self.tuned_learners_ensemble.evaluate_iterator(self.test_iterator)
    #     else:
    #         train_loss, train_acc = self.learners_ensemble.evaluate_iterator(self.val_iterator)
    #         test_loss, test_acc = self.learners_ensemble.evaluate_iterator(self.test_iterator)
    #         # train_recon, train_nll = self.learners_ensemble.evaluate_ac_iterator(self.val_iterator)
    #         # test_recon, test_nll = self.learners_ensemble.evaluate_ac_iterator(self.test_iterator)

    #     train_recon = 0
    #     train_nll = 0
    #     test_recon = 0
    #     test_nll = 0
    #     #not used
    #     self.logger.add_scalar("Train/Loss", train_loss, self.counter)
    #     self.logger.add_scalar("Train/Metric", train_acc, self.counter)
    #     self.logger.add_scalar("Test/Loss", test_loss, self.counter)
    #     self.logger.add_scalar("Test/Metric", test_acc, self.counter)

    #     self.logger.add_scalar("Train/Recon_Loss", train_recon, self.counter)
    #     self.logger.add_scalar("Train/NLL", train_nll, self.counter)
    #     self.logger.add_scalar("Test/Recon_Loss", test_recon, self.counter)
    #     self.logger.add_scalar("Test/NLL", test_nll, self.counter)

    #     return train_loss, train_acc, test_loss, test_acc, train_recon, train_nll, test_recon, test_nll
