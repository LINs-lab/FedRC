import torch
import torch.nn.functional as F
from utils.losses import BinaryClassifierLoss


class Learner:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    is_binary_classification (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True

    Methods
    ------
    compute_gradients_and_loss:

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_epoch:

    fit_batches: perform successive optimizer steps over successive batches

    fit_epochs:

    evaluate_iterator: evaluate `model` on an iterator

    gather_losses:

    get_param_tensor: get `model` parameters as a unique flattened tensor

    free_memory: free the memory allocated by the model weights

    free_gradients:
    """

    def __init__(
            self, model,
            criterion,
            metric,
            device,
            optimizer,
            lr_scheduler=None,
            is_binary_classification=False,
            penalty = None,
            phi_model = None
    ):

        self.model = model
        self.phi_model = model
        self.criterion = criterion.to(device)
        self.metric = metric
        # self.penalty = BinaryClassifierLoss_Negative(class_number=self.criterion.class_number)
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_binary_classification = is_binary_classification
        # self.labels_weights = torch.ones(self.criterion.class_number) / self.criterion.class_number

        self.model_dim = int(self.get_param_tensor().shape[0])
        # self.grad_dim = int(self.get_grad_tensor().shape[0])

    def optimizer_step(self):
        """
         perform one optimizer step, requires the gradients to be already computed.
        """
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        self.model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        return loss.detach()

    def fit_batch(self, batch, weights=None):
        """
        perform an optimizer step over one batch drawn from `iterator`

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()
        self.model.to(self.device)

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y) / len(y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        self.model.to('cpu')

        return loss.detach(), metric.detach()

    def diverse_criterion(self, preds, other_preds, targets):
        label_dist = torch.tensor([self.labels_weights[target] for target in targets]).to(self.device)
        if len(other_preds) == 0:
            return 0
        # L = [F.kl_div(preds, other_pred) for other_pred in other_preds]
        L = [F.kl_div(torch.exp(-F.cross_entropy(preds, targets) - torch.log(label_dist)) , torch.exp(-F.cross_entropy(other_pred, targets) - torch.log(label_dist))) for other_pred in other_preds]
        return - sum(L) / len(other_preds)

    def fit_epoch_diverse(self, iterator, weights=None, learners=None, learner_id=None):
        if not learners:
            return self.fit_epoch(iterator, weights=weights)

        self.model.train()
        self.model.to(self.device)
        for learner_id, learner in enumerate(learners):
            learner.model.to(self.device)

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)
            # print(y)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)
            else:
                y = y.type(torch.long)

            self.optimizer.zero_grad()

            y_pred = self.model(x)
            other_learners_preds = [learners[i].model(x).clone().detach().to(self.device) for i in range(len(learners)) if i != learner_id]

            loss_vec = self.criterion(y_pred, y)
            diverse_loss_vec = self.diverse_criterion(y_pred, other_learners_preds, y)
            loss_vec = loss_vec + 0.1 * diverse_loss_vec
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).detach()

            x.to('cpu')
            y.to('cpu')
        
        self.model.to('cpu')
        for learner_id, learner in enumerate(learners):
            learner.model.to('cpu')

        return global_loss / n_samples, global_metric / n_samples

    def feddecorr_loss(self, z):
        N,d = z.shape
        # z-score normalization
        # print(z.std(0))
        z = (z - z.mean(0)) / (z.std(0) + 1e-4)
        # estimate correlation matrix
        corr_mat = 1/N*torch.matmul(z.t(), z)
        # calculate FedDecorr loss
        loss_fed_decorr = (corr_mat.pow(2)).mean()
        return loss_fed_decorr



    def fit_epoch(self, iterator, weights=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()
        self.model.to(self.device)

        global_loss = 0.
        global_metric = 0.
        n_samples = 1
        count_phi = 0
        loss_vec_phi = 0.0
        client_representation = None

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)



            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)
            else:
                y = y.type(torch.long)

            self.optimizer.zero_grad()

            y_pred = self.model(x)


            loss_vec = self.criterion(y_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            # z = self.model.features(x)
            # # print(z.squeeze().shape)
            # loss += 0.1 * self.feddecorr_loss(z.squeeze())

            
            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).detach()

            x.to('cpu')
            y.to('cpu')
        
        self.model.to('cpu')




        if n_samples == 0:
            return 0.0, 1.0

        return global_loss / n_samples, global_metric / n_samples, client_representation

    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        self.model.to(self.device)
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)
                else:
                    y = y.type(torch.long)

                y_pred = self.model(x)
                # print(y)
                if isinstance(self.criterion, BinaryClassifierLoss):
                    all_losses[indices] = self.criterion(y_pred, y, positive_only=True).squeeze()
                else:
                    all_losses[indices] = self.criterion(y_pred, y).squeeze()

        self.model.to('cpu')
        return all_losses

    def gather_losses_split(self, base_model, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        self.model.to(self.device)
        base_model.to(self.device)
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)
                else:
                    y = y.type(torch.long)

                y_pred = self.model(base_model(x))
                # print(y)
                if isinstance(self.criterion, BinaryClassifierLoss):
                    all_losses[indices] = self.criterion(y_pred, y, positive_only=True).squeeze()
                else:
                    all_losses[indices] = self.criterion(y_pred, y).squeeze()

        self.model.to('cpu')
        base_model.to('cpu')
        return all_losses


    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()
        self.model.to(self.device)

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, _ in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            with torch.no_grad():
                y_pred = self.model(x)

                global_loss += self.criterion(y_pred, y).sum().detach()
                global_metric += self.metric(y_pred, y).detach()

            n_samples += y.size(0)

        self.model.to('cpu')

        return global_loss / n_samples, global_metric / n_samples

    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch(iterator, weights)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        client_representation = None
        if self.phi_model is not None:
            loss_vec_phi = 0.0
            self.phi_model.train()
            self.phi_model.to(self.device)
            n_samples = 0

            for x, y, indices in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)
                n_samples += y.size(0)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)
                else:
                    y = y.type(torch.long)

                y_pred_phi = self.phi_model(x)
                loss_vec_phi += torch.sum(self.criterion(y_pred_phi, y))
            loss_vec_phi = loss_vec_phi / n_samples
            loss_vec_phi.backward()
            client_representation = self.get_grad_tensor_model(self.phi_model)
            self.phi_model.zero_grad()
            self.phi_model.to('cpu')
            self.phi_model = None

        return client_representation

    def fit_epoch_split(self, iterator, base_model, weights=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()
        self.model.to(self.device)
        base_model.eval()
        base_model.to(self.device)

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)
            # print(y)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)
            else:
                y = y.type(torch.long)

            self.optimizer.zero_grad()

            # y_pred = self.model(base_model(x))
            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y) 
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).detach()

            x.to('cpu')
            y.to('cpu')
        
        self.model.to('cpu')
        base_model.to('cpu')

        return global_loss / n_samples, global_metric / n_samples

    def fit_epochs_split(self, iterator, n_epochs, base_model, weights=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch_split(iterator, base_model, weights)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def fit_epoch_base(self, iterator, learners, weights=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()
        self.model.to(self.device)
        for learner in learners:
            learner.model.eval()
            learner.model.to(self.device)


        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)
            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)
            else:
                y = y.type(torch.long)

            self.optimizer.zero_grad()

            x_features = self.model(x)

            
                    
            loss = 0.0
            for learner_id, learner in enumerate(learners):      
                y_pred = learner.model(x_features)
                loss_vec = self.criterion(y_pred, y)
                if weights is not None:
                    weights_i = weights[learner_id].to(self.device)
                    loss += (loss_vec.T @ weights_i[indices]) / loss_vec.size(0)
                else:
                    loss += loss_vec.mean()
            loss.backward()

            self.optimizer.step()

            # global_loss += loss.detach() * loss_vec.size(0)
            # global_metric += self.metric(y_pred, y).detach()

            x.to('cpu')
            y.to('cpu')
        
        self.model.to('cpu')
        for learner in learners:
            # learner.model.eval()
            learner.model.to('cpu')

        # return global_loss / n_samples, global_metric / n_samples

    def fit_epoch_disc(self, iterator, base_learner, weights=None):

        self.model.train()
        self.model.to(self.device)
        base_learner.model.eval()
        base_learner.model.to(self.device)

        weights = weights.to(self.device)

        # disc_loss = 0.0
        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)
            else:
                y = y.type(torch.long)

            self.optimizer.zero_grad()

            x_features = base_learner.model(x)

            disc_pred = self.model(x_features)
            # disc_loss = - torch.mean(torch.sum(F.log_softmax(disc_pred, dim=1) * weights.T[indices, :], dim=1))
            disc_loss = torch.mean(torch.sum(torch.abs(F.softmax(disc_pred, dim=1) - weights.T[indices, :]), dim=1))

            disc_loss.backward()
            self.optimizer.step()

            x.to('cpu')
            y.to('cpu')

        # disc_loss = disc_loss / len(iterator)
        # disc_loss.backward()

        # self.optimizer.step()

            # global_loss += loss.detach() * loss_vec.size(0)
            # global_metric += self.metric(y_pred, y).detach()

        
        self.model.to('cpu')
        base_learner.model.to('cpu')

        weights = weights.to('cpu')




    def fit_epoch_base_disc(self, iterator, learners, domain_disc_learner, weights=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()
        self.model.to(self.device)
        for learner in learners:
            learner.model.eval()
            learner.model.to(self.device)
        domain_disc_learner.model.eval()
        domain_disc_learner.model.to(self.device)

        weights = weights.to(self.device)


        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)
            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)
            else:
                y = y.type(torch.long)

            self.optimizer.zero_grad()

            x_features = self.model(x)

            
                    
            loss = 0.0
            for learner_id, learner in enumerate(learners):      
                y_pred = learner.model(x_features)
                loss_vec = self.criterion(y_pred, y)
                if weights is not None:
                    weights_i = weights[learner_id].to(self.device)
                    loss += (loss_vec.T @ weights_i[indices]) / loss_vec.size(0)
                else:
                    loss += loss_vec.mean()

            disc_pred = domain_disc_learner.model(x_features)
            # disc_loss = - torch.mean(torch.sum(F.log_softmax(disc_pred, dim=1) * weights.T[indices, :], dim=1))
            disc_loss = torch.mean(torch.sum(torch.abs(F.softmax(disc_pred, dim=1) - weights.T[indices, :]), dim=1))

            # alpha = 0.1 * loss.item() / disc_loss.item()
            alpha = 0.1


            loss = loss - alpha * disc_loss

            # print(disc_loss)

            loss.backward()

            self.optimizer.step()

            # global_loss += loss.detach() * loss_vec.size(0)
            # global_metric += self.metric(y_pred, y).detach()

            x.to('cpu')
            y.to('cpu')
        
        self.model.to('cpu')
        for learner in learners:
            # learner.model.eval()
            learner.model.to('cpu')
        domain_disc_learner.model.to('cpu')
        weights = weights.to('cpu')

    def fit_epochs_disc(self, iterator, n_epochs, base_learner, weights=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch_disc(iterator, base_learner, weights)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def fit_epochs_base(self, iterator, n_epochs, learners, weights=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch_base(iterator, learners, weights)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
    def fit_epochs_base_disc(self, iterator, n_epochs, learners, domain_disc_learner, weights=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch_base_disc(iterator, learners, domain_disc_learner, weights)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)
    
    def get_grad_tensor_model(self, model):
        grad_list = []

        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def free_memory(self):
        """
        free the memory allocated by the model weights

        """
        del self.optimizer
        del self.model

    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        self.optimizer.zero_grad(set_to_none=True)


class LanguageModelingLearner(Learner):
    def fit_epoch(self, iterator, weights=None):

        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device)
            y = y.to(self.device)

            n_samples += y.size(0)

            chunk_len = y.size(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)
            loss_vec = self.criterion(y_pred, y)

            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0) / chunk_len
            global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def fit_batch(self, batch, weights=None):

        self.model.train()

        x, y, indices = batch
        x = x.to(self.device)
        y = y.to(self.device)

        n_samples = y.size(0)
        chunk_len = y.size(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()

        global_loss = loss.detach() * loss_vec.size(0) / chunk_len
        global_metric = self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        raise NotImplementedError

    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        # predictions = torch.zeros(n_samples, device=self.device)
        predictions = torch.tensor([], device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
                # print(x, y_pred)
                predictions[indices] = self.criterion(y_pred, y).mean(axis=1)
                # predictions = torch.cat((predictions, self.criterion(y_pred, y).reshape(len(y) * 80)), 0)
                # print(predictions, self.criterion(y_pred, y))

        return predictions

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y, _ in iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                n_samples += y.size(0)

                chunk_len = y.size(1)

                y_pred = self.model(x)
                global_loss += self.criterion(y_pred, y).sum().detach() / chunk_len
                global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples
