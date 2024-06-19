from copy import copy
import time
from sklearn.decomposition import PCA
from resnet import *

from models import *
from datasets import *
from learners.learner import *
from learners.learners_ensemble import *
from learners.autoencoder import *
from client import *
from aggregator import *

from .optim import *
from .metrics import *
from .constants import *
from .decentralized import *
from .losses import *

from torch.utils.data import DataLoader

from tqdm import tqdm


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir

def get_split_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        init_model = None,
        binary=False
):
    if name == 'fmnist-c':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN_Classifier(num_classes=10)
        is_binary_classification = False
    elif name == "cifar10" or 'cifar10-c' in name:
        # criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        if binary:
            criterion = BinaryClassifierLoss(class_number=10).to(device)
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        # model = get_mobilenet_classifier(n_classes=10)
        model = get_resnet_classifier(n_classes=10)
        is_binary_classification = False
    elif name == "cifar100" or 'cifar100-c' in name or name == 'cifar100-c-10':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet_classifier(n_classes=100)
        # model = get_resnet_classifier(n_classes=100)
        # model = get_resnet18(n_classes=100)
        is_binary_classification = False
    elif name == "tiny-imagenet-c" or 'tiny-imagenet-c' in name:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet_classifier(n_classes=200)
        # model = get_resnet_classifier(n_classes=10)
        is_binary_classification = False

    else:
        raise NotImplementedError

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu
        )
    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    if name == "shakespeare":
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    else:
        return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )

def get_domain_discriminator(name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        num_domains,
        input_dim=None,
        output_dim=None):

    if name == 'fmnist-c':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN_Classifier(num_classes=num_domains)
        is_binary_classification = False
    elif name == "cifar10" or 'cifar10-c' in name:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet_classifier(n_classes=num_domains)
        # model = get_resnet_classifier(n_classes=num_domains)
        is_binary_classification = False
    elif name == "cifar100" or name == 'cifar100-c' or name == 'cifar100-c-10':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet_classifier(n_classes=num_domains)
        # model = get_resnet_classifier(n_classes=num_domains)
        # model = get_resnet18(n_classes=100)
        is_binary_classification = False
    elif name == "tiny-imagenet-c" or 'tiny-imagenet-c' in name:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet_classifier(n_classes=num_domains)
        # model = get_resnet_classifier(n_classes=10)
        is_binary_classification = False

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu
        )
    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )

def get_base_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        init_model = None,
        binary=False
):
    if name == 'fmnist-c':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN_Feature(num_classes=10)
        is_binary_classification = False
    elif name == "cifar10" or 'cifar10-c' in name:
        # criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        if binary:
            criterion = BinaryClassifierLoss(class_number=10).to(device)
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        # model = get_mobilenet_feature(n_classes=10)
        model = get_resnet_feature(n_classes=10)
        is_binary_classification = False
    elif name == "cifar100" or 'cifar100-c' in name or name == 'cifar100-c-10':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet_feature(n_classes=100)
        # model = get_resnet_feature(n_classes=100)
        # model = get_resnet18(n_classes=100)
        is_binary_classification = False
    elif name == "tiny-imagenet-c" or 'tiny-imagenet-c' in name:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet_feature(n_classes=200)
        # model = get_resnet_classifier(n_classes=10)
        is_binary_classification = False

    else:
        raise NotImplementedError

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            # lr_initial=min(initial_lr, 0.06),
            lr_initial=initial_lr,
            mu=mu
        )
    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    if name == "shakespeare":
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    else:
        return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )


def get_split_learners_ensemble(
        n_learners,
        client_type,
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        hard_cluster,
        n_rounds,
        seed,
        n_gmm,
        domain_disc=False,
        input_dim=None,
        output_dim=None,
        embedding_dim=None,
        binary=False
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param n_learners: number of learners in the ensemble
    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: LearnersEnsemble

    """
    learners = [
        get_split_learner(
            name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed + learner_id,
            mu=mu,
            binary=binary
        ) for learner_id in range(n_learners)
    ]

    base_learner = get_base_learner(name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed -1,
            mu=mu,
            binary=binary)

    if domain_disc:
        domain_disc_learner = get_domain_discriminator(name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed -1,
            mu=mu,
            num_domains=n_learners)
    else:
        domain_disc_learner = None

    learners_weights = torch.ones(n_learners) / n_learners
    if client_type == "ACGmixture":
        if name == "mnist" or name == "emnist" or name == "femnist" or name == "mnist9" or name == "emnist_r":
            assert embedding_dim is not None, "Embedding dimension not specified!!"
            model = resnet_pca(embedding_size=embedding_dim, name=name,input_size=(1, 28, 28))
            ckpt = 'AE_emnist.pt'
            if name == "mnist9":
                ckpt = 'AE_MNIST1.pt'
            if global_ac == None:
                global_ac = Autoencoder(
                    model=model,
                    checkpoint=None,
                    criterion=torch.nn.BCELoss(reduction='none'),
                    device=learners[0].device,
                    optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                    lr_scheduler=None
                )
                # global_ac  = models.resnet50(pretrained=True)
                global_ac.freeze()
            # ac = Autoencoder(
            #     model=model,
            #     checkpoint='AE_emnist.pt',
            #     device=learners[0].device,
            #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
            #     lr_scheduler=None
            # )
            ac = global_ac
            return ACGLearnersEnsemble(
                learners=learners,
                embedding_dim=embedding_dim,
                autoencoder=ac,
                n_gmm=n_gmm
            )
        elif name == 'synthetic' or name == 'gmsynthetic':
            assert embedding_dim is not None, "Embedding dimension not specified!!"
            model = IDnetwork(embedding_size=input_dim)
            if global_ac == None:
                global_ac = Autoencoder(
                    model=model,
                    checkpoint=None,
                    criterion=torch.nn.MSELoss(reduction='none'),
                    device=learners[0].device,
                    optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                    lr_scheduler=None
                )
                # global_ac = models.resnet50(pretrained=True)
                global_ac.freeze()
            # ac = Autoencoder(
            #     model=model,
            #     checkpoint='AE_emnist.pt',
            #     device=learners[0].device,
            #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
            #     lr_scheduler=None
            # )
            ac = global_ac
            return ACGLearnersEnsemble(
                learners=learners,
                embedding_dim=embedding_dim,
                autoencoder=ac,
                n_gmm=n_gmm
            )
        elif name == "cifar10" or name == 'cifar100':
            # Resnet_PCA
            # model = Resnet_PCA(embedding_size=embedding_dim, input_size=(3, 32, 32))
            # model = resnet_pca(embedding_size=embedding_dim, name=name, input_size=(3, 32, 32))
            # model = cACnetwork(embedding_size=embedding_dim, input_size=(3, 32, 32))
            model = resnet_pca(embedding_size=embedding_dim, name=name, input_size=(3, 32, 32))
            if global_ac == None:
                global_ac = Autoencoder(
                    model=model,
                    # checkpoint='AE_CIFAR10.pt',
                    checkpoint=None,
                    criterion=torch.nn.BCELoss(reduction='none'),
                    device=learners[0].device,
                    optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                    lr_scheduler=None
                )
                # global_ac = Autoencoder(
                #     model=model,
                #     checkpoint=None,
                #     criterion=torch.nn.BCELoss(reduction='none'),
                #     device=learners[0].device,
                #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                #     lr_scheduler=None
                # )
                # global_ac = models.resnet50(pretrained=True)
                global_ac.freeze()
            # ac = Autoencoder(
            #     model=model,
            #     checkpoint='AE_emnist.pt',
            #     device=learners[0].device,
            #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
            #     lr_scheduler=None
            # )
            ac = global_ac
            return ACGLearnersEnsemble(
                learners=learners,
                embedding_dim=embedding_dim,
                autoencoder=ac,
                n_gmm=n_gmm
            )
        else:
            raise NotImplementedError('Experiment setting not implemented yet.')
    else:
        if name == "shakespeare":
            return LanguageModelingLearnersEnsemble(learners=learners, learners_weights=learners_weights)
        else:
            return SplitLearnersEnsemble(base_learner=base_learner,learners=learners, learners_weights=learners_weights, device=device,domain_disc_learner=domain_disc_learner,hard_cluster=hard_cluster)


def get_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        init_model = None,
        binary=False,
        phi_model=False
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: Learner

    """
    torch.manual_seed(seed)

    if name == "synthetic":
        if output_dim == 2:
            criterion = BinaryClassifierLoss(class_number=10).to(device)
            metric = binary_accuracy
            model = LinearLayer(input_dim, 1).to(device)
            is_binary_classification = True
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
            metric = accuracy
            model = LinearLayer(input_dim, output_dim).to(device)
            is_binary_classification = False
    elif name == 'fmnist-c':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=10)
        is_binary_classification = False
    elif name == 'airline':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = Re_MLP(num_classes=2, in_num=12,mid_num=128)
        is_binary_classification = False
    elif name == 'elec':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = Re_MLP(num_classes=2, in_num=6,mid_num=12)
        is_binary_classification = False
    elif name == 'powersupply':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = Re_MLP(num_classes=2, in_num=3,mid_num=6)
        is_binary_classification = False
    elif name == "cifar10" or 'cifar10-c' in name:
        # criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        if binary:
            criterion = BinaryClassifierLoss(class_number=10).to(device)
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        if not init_model:
            # model = get_mobilenet(n_classes=10)
            model = get_resnet18(n_classes=10)
        else:
            model = init_model
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        # model = get_resnet18(n_classes=10).to(device)
        is_binary_classification = False
    elif name == "cifar100" or 'cifar100-c' in name or name == 'cifar100-c-10':
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet(n_classes=100)
        # model = get_resnet18(n_classes=100)
        is_binary_classification = False
    elif name == "tiny-imagenet-c" or 'tiny-imagenet-c' in name:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet(n_classes=200)
        # model = get_resnet_classifier(n_classes=10)
        is_binary_classification = False
    elif name == "emnist" or name == "femnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=62).to(device)
        is_binary_classification = False
    elif name == "shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8

        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
        metric = accuracy
        model =\
            NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"]
            ).to(device)
        is_binary_classification = False

    else:
        raise NotImplementedError

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu
        )
    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    if name == "shakespeare":
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    elif phi_model:
        return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification,
            phi_model = deepcopy(model)
        )
    else:
        return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification,
            phi_model = None
        )



def get_learners_ensemble(
        n_learners,
        client_type,
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        hard_cluster,
        n_rounds,
        seed,
        n_gmm,
        domain_disc=False,
        input_dim=None,
        output_dim=None,
        embedding_dim=None,
        binary=False,
        phi_model=False
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param n_learners: number of learners in the ensemble
    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: LearnersEnsemble

    """
    learners = [
        get_learner(
            name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed + learner_id,
            mu=mu,
            binary=binary,
            phi_model=phi_model
        ) for learner_id in range(n_learners)
    ]

    learners_weights = torch.ones(n_learners) / n_learners
    if client_type == "ACGmixture":
        if name == "mnist" or name == "emnist" or name == "femnist" or name == "mnist9" or name == "emnist_r":
            assert embedding_dim is not None, "Embedding dimension not specified!!"
            model = resnet_pca(embedding_size=embedding_dim, name=name,input_size=(1, 28, 28))
            ckpt = 'AE_emnist.pt'
            if name == "mnist9":
                ckpt = 'AE_MNIST1.pt'
            if global_ac == None:
                global_ac = Autoencoder(
                    model=model,
                    checkpoint=None,
                    criterion=torch.nn.BCELoss(reduction='none'),
                    device=learners[0].device,
                    optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                    lr_scheduler=None
                )
                # global_ac  = models.resnet50(pretrained=True)
                global_ac.freeze()
            # ac = Autoencoder(
            #     model=model,
            #     checkpoint='AE_emnist.pt',
            #     device=learners[0].device,
            #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
            #     lr_scheduler=None
            # )
            ac = global_ac
            return ACGLearnersEnsemble(
                learners=learners,
                embedding_dim=embedding_dim,
                autoencoder=ac,
                n_gmm=n_gmm
            )
        elif name == 'synthetic' or name == 'gmsynthetic':
            assert embedding_dim is not None, "Embedding dimension not specified!!"
            model = IDnetwork(embedding_size=input_dim)
            if global_ac == None:
                global_ac = Autoencoder(
                    model=model,
                    checkpoint=None,
                    criterion=torch.nn.MSELoss(reduction='none'),
                    device=learners[0].device,
                    optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                    lr_scheduler=None
                )
                # global_ac = models.resnet50(pretrained=True)
                global_ac.freeze()
            # ac = Autoencoder(
            #     model=model,
            #     checkpoint='AE_emnist.pt',
            #     device=learners[0].device,
            #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
            #     lr_scheduler=None
            # )
            ac = global_ac
            return ACGLearnersEnsemble(
                learners=learners,
                embedding_dim=embedding_dim,
                autoencoder=ac,
                n_gmm=n_gmm
            )
        elif "cifar10" in name or 'cifar100' in name:
            # Resnet_PCA
            # model = Resnet_PCA(embedding_size=embedding_dim, input_size=(3, 32, 32))
            # model = resnet_pca(embedding_size=embedding_dim, name=name, input_size=(3, 32, 32))
            # model = cACnetwork(embedding_size=embedding_dim, input_size=(3, 32, 32))
            model = resnet_pca(embedding_size=embedding_dim, name=name, input_size=(3, 32, 32))
            if global_ac == None:
                global_ac = Autoencoder(
                    model=model,
                    # checkpoint='AE_CIFAR10.pt',
                    checkpoint=None,
                    criterion=torch.nn.BCELoss(reduction='none'),
                    device=learners[0].device,
                    optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                    lr_scheduler=None
                )
                # global_ac = Autoencoder(
                #     model=model,
                #     checkpoint=None,
                #     criterion=torch.nn.BCELoss(reduction='none'),
                #     device=learners[0].device,
                #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                #     lr_scheduler=None
                # )
                # global_ac = models.resnet50(pretrained=True)
                global_ac.freeze()
            # ac = Autoencoder(
            #     model=model,
            #     checkpoint='AE_emnist.pt',
            #     device=learners[0].device,
            #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
            #     lr_scheduler=None
            # )
            ac = global_ac
            return ACGLearnersEnsemble(
                learners=learners,
                embedding_dim=embedding_dim,
                autoencoder=ac,
                n_gmm=n_gmm
            )
        else:
            raise NotImplementedError('Experiment setting not implemented yet.')
    else:
        if name == "shakespeare":
            return LanguageModelingLearnersEnsemble(learners=learners, learners_weights=learners_weights)
        else:
            return LearnersEnsemble(learners=learners, learners_weights=learners_weights,device=device)

def split_train_val_test(data_size, portion=0.9):

    random_split = [i for i in range(data_size)]
    random.shuffle(random_split)
    train_size = int(portion * data_size)
    return random_split[:train_size], random_split[:int(train_size)], random_split[train_size:]

def get_fmnistC_loaders(root_path, batch_size, is_validation, test=False):
    train_iterators, val_iterators, test_iterators, client_types,client_features = [], [], [], [], []
    if test:
        for i in range(1, 4):
            task_data_path = os.path.join(root_path, 'test-{}.pkl'.format(i))
            with open(task_data_path, "rb") as f:
                client = pickle.load(f)
            train_indices, val_indices, test_indices = split_train_val_test(len(client['labels']))
            train_dataset = SubFEMNISTC(train_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
            val_dataset = SubFEMNISTC(val_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
            test_dataset = SubFEMNISTC(test_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
            train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)
            client_types.append(client['type'])
            client_features.append([0] * len(client['labels']))

        return train_iterators, val_iterators, test_iterators, client_types, client_features

    for i in range(300):
        task_data_path = os.path.join(root_path, '{}.pkl'.format(i))
        with open(task_data_path, "rb") as f:
            client = pickle.load(f)
        train_indices, val_indices, test_indices = split_train_val_test(len(client['labels']))
        train_dataset = SubFEMNISTC(train_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
        val_dataset = SubFEMNISTC(val_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
        test_dataset = SubFEMNISTC(test_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
        train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)
        client_types.append(client['type'])
        client_features.append([0] * len(client['labels']))

    return train_iterators, val_iterators, test_iterators, client_types,client_features

def get_airline_loaders(root_path, batch_size, is_validation, test=False):
    train_iterators, val_iterators, test_iterators, client_types, client_features = [], [], [], [], []
    if test:
        for i in range(2):
            task_data_path = os.path.join(root_path, 'test-{}.pkl'.format(i))
            with open(task_data_path, "rb") as f:
                client = pickle.load(f)
            train_indices, val_indices, test_indices = split_train_val_test(len(client['labels']), portion=0.4)
            train_dataset = SubPowerSupply(train_indices, cifar10_data=client['data'], cifar10_targets=client['labels'])
            val_dataset = SubPowerSupply(val_indices, cifar10_data=client['data'], cifar10_targets=client['labels'])
            test_dataset = SubPowerSupply(test_indices, cifar10_data=client['data'], cifar10_targets=client['labels'])
            train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)
            client_types.append(0)
            client_features.append([0] * len(client['labels']))

        return train_iterators, val_iterators, test_iterators, client_types, client_features

    for i in range(300):
        task_data_path = os.path.join(root_path, '{}.pkl'.format(i))
        with open(task_data_path, "rb") as f:
            client = pickle.load(f)
        train_indices, val_indices, test_indices = split_train_val_test(len(client['labels']))
        train_dataset = SubPowerSupply(train_indices, cifar10_data=client['data'], cifar10_targets=client['labels'])
        val_dataset = SubPowerSupply(val_indices, cifar10_data=client['data'], cifar10_targets=client['labels'])
        test_dataset = SubPowerSupply(test_indices, cifar10_data=client['data'], cifar10_targets=client['labels'])
        train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)
        client_types.append(0)
        client_features.append([0] * len(client['labels']))

    return train_iterators, val_iterators, test_iterators, client_types, client_features

def get_imagenetC_loaders(root_path, batch_size, is_validation, test=False, test_num = 3):
    train_iterators, val_iterators, test_iterators, client_types, client_features = [], [], [], [], []
    if test:
        for i in range(1, test_num + 1):
            task_data_path = os.path.join(root_path, 'test-{}.pkl'.format(i))
            with open(task_data_path, "rb") as f:
                client = pickle.load(f)
            train_indices, val_indices, test_indices = split_train_val_test(len(client['labels']))
            train_dataset = SubImagenetC(train_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
            val_dataset = SubImagenetC(val_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
            test_dataset = SubImagenetC(test_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
            train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)
            client_types.append(client['type'])
            if 'features' in client:
                client_features.append(client['features'])
            else:
                client_features.append([0] * len(client['labels']))

        return train_iterators, val_iterators, test_iterators, client_types, client_features

    for i in range(100):
        task_data_path = os.path.join(root_path, '{}.pkl'.format(i))
        with open(task_data_path, "rb") as f:
            client = pickle.load(f)
        train_indices, val_indices, test_indices = split_train_val_test(len(client['labels']))
        train_dataset = SubImagenetC(train_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
        val_dataset = SubImagenetC(val_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
        test_dataset = SubImagenetC(test_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
        train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)
        client_types.append(client['type'])
        if 'features' in client:
            client_features.append(client['features'])
        else:
            client_features.append([0] * len(client['labels']))

    return train_iterators, val_iterators, test_iterators, client_types, client_features

def get_cifar10C_loaders(root_path, batch_size, is_validation, test=False, test_num = 3, train_num=100):
    train_iterators, val_iterators, test_iterators, client_types, client_features = [], [], [], [], []
    if test:
        for i in range(1, test_num + 1):
            task_data_path = os.path.join(root_path, 'test-{}.pkl'.format(i))
            with open(task_data_path, "rb") as f:
                client = pickle.load(f)
            train_indices, val_indices, test_indices = split_train_val_test(len(client['labels']))
            train_dataset = SubCIFAR10C(train_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
            val_dataset = SubCIFAR10C(val_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
            test_dataset = SubCIFAR10C(test_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
            train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)
            client_types.append(client['type'])
            if 'features' in client:
                client_features.append(client['features'])
            else:
                client_features.append([0] * len(client['labels']))

        return train_iterators, val_iterators, test_iterators, client_types, client_features

    for i in range(train_num):
        task_data_path = os.path.join(root_path, '{}.pkl'.format(i))
        with open(task_data_path, "rb") as f:
            client = pickle.load(f)
        train_indices, val_indices, test_indices = split_train_val_test(len(client['labels']))
        train_dataset = SubCIFAR10C(train_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
        val_dataset = SubCIFAR10C(val_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
        test_dataset = SubCIFAR10C(test_indices, cifar10_data=client['images'], cifar10_targets=client['labels'])
        train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)
        client_types.append(client['type'])
        if 'features' in client:
                client_features.append(client['features'])
        else:
            client_features.append([0] * len(client['labels']))

    return train_iterators, val_iterators, test_iterators, client_types, client_features


def get_loaders(type_, root_path, batch_size, is_validation):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last

    :param type_: type of the dataset;
    :param root_path: path to the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])

    """
    if type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    elif type_ == "emnist":
        inputs, targets = get_emnist()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators, client_types, feature_types = [], [], [], [], []

    for task_id, task_dir in enumerate(tqdm(os.listdir(root_path))):
        task_data_path = os.path.join(root_path, task_dir)

        train_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True
            )

        val_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )
        
        if not train_iterator or not test_iterator:
            continue

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)
        client_types.append(task_id)
        feature_types.append([0] * len(train_iterator.dataset))

    return train_iterators, val_iterators, test_iterators, client_types, feature_types


def get_loader(type_, path, batch_size, train, inputs=None, targets=None):
    """
    constructs a torch.utils.DataLoader object from the given path

    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader

    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets)
    elif type_ == "emnist":
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    # print(len(dataset))
    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)


def get_client(
        client_type,
        learners_ensemble,
        q,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally,
        data_type = 0,
        feature_type = None,
        class_number = 10
):
    """

    :param client_type:
    :param learners_ensemble:
    :param q: fairness hyper-parameter, ony used for FFL client
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param tune_locally

    :return:

    """
    if client_type == "mixture":
        return MixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "mixture_SW":
        return MixtureClient_SW(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "fedrc":
        return FedRC(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "fedrc_tune":
        return FedRC(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "fedrc_Adam":
        return FedRC_Adam(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "fedrc_SW":
        return FedRC_SW(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "ACGmixture":
        return ACGMixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "fedrc_DP":
        return FedRC_DP(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "IFCA":
        return IFCA(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "FeSEM":
        return FeSEM(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "AFL":
        return AgnosticFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "FFL":
        return FFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            q=q,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    elif client_type == "FedSoft":
        return FedSoft(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )
    else:
        return Client(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            data_type=data_type,
            feature_types=feature_type,
            class_number = class_number
        )


def get_aggregator(
        aggregator_type,
        clients,
        global_learners_ensemble,
        lr,
        lr_lambda,
        mu,
        communication_probability,
        q,
        sampling_rate,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        verbose,
        seed=None,
        experiment=None,
        method=None,
        suffix=None,
        split=False,
        domain_disc=False,
        em_step=1
):
    """
    `personalized` corresponds to pFedMe

    :param aggregator_type:
    :param clients:
    :param global_learners_ensemble:
    :param lr: oly used with FLL aggregator
    :param lr_lambda: only used with Agnostic aggregator
    :param mu: penalization term, only used with L2SGD
    :param communication_probability: communication probability, only used with L2SGD
    :param q: fairness hyper-parameter, ony used for FFL client
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "ACGcentralized":
        return ACGCentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            em_step=em_step,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "FedIAS":
        return FedIASAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "conceptem_ts":
        return FedRCTSAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "IFCA":
        return IFCAAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "FeSEM":
        return FeSEMAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "FedSoft":
        return FedSoftAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "personalized":
        return PersonalizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "APFL":
        return APFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            alpha=0.5,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "clustered":
        return ClusteredAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "STOCFLAggregator":
        return STOCFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "ICFLAggregator":
        return ICFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "L2SGD":
        return LoopLessLocalSGDAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            communication_probability=communication_probability,
            penalty_parameter=mu,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "AFL":
        return AgnosticAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr_lambda=lr_lambda,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "FFL":
        return FFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr=lr,
            q=q,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    elif aggregator_type == "decentralized":
        n_clients = len(clients)
        mixing_matrix = get_mixing_matrix(n=n_clients, p=0.5, seed=seed)

        return DecentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            mixing_matrix=mixing_matrix,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            experiment=experiment,
            method=method,
            suffix=suffix,
            split=split,
            domain_disc=domain_disc
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized`,"
            " `personalized`, `clustered`, `fednova`, `AFL`,"
            " `FFL` and `decentralized`."
        )
