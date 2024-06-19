"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
from sklearn import cluster
from utils.utils import *
from utils.constants import *
from utils.args import *

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_, root_path, logs_dir):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]
    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else:
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        if args_.split:
            learners_ensemble =\
            get_split_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                domain_disc=args_.domain_disc,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary
            )
        else:
            learners_ensemble =\
                get_learners_ensemble(
                    n_learners=args_.n_learners,
                    client_type=CLIENT_TYPE[args_.method],
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    input_dim=args_.input_dimension,
                    output_dim=args_.output_dimension,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    mu=args_.mu,
                    n_gmm=args_.n_gmm,
                    embedding_dim=args_.embedding_dimension,
                    hard_cluster=args_.hard_cluster,
                    binary=args_.binary,
                    phi_model=args.phi_model
                )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        if CLIENT_TYPE[args_.method] == "conceptEM_tune" and "train" in logs_dir:

            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=True,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )
        else:
            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=args_.locally_tune_clients,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )

        clients_.append(client)

    return clients_


def run_experiment(args_):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients = init_clients(args_, root_path=os.path.join(data_dir, "train"), logs_dir=os.path.join(logs_dir, "train"))

    print("==> Test Clients initialization..")
    test_clients = init_clients(args_, root_path=os.path.join(data_dir, "test"),
                                logs_dir=os.path.join(logs_dir, "test"))

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    if args_.split:
        global_learners_ensemble = \
        get_split_learners_ensemble(
            n_learners=args_.n_learners,
            client_type=CLIENT_TYPE[args_.method],
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            embedding_dim=args_.embedding_dimension,
            n_gmm=args_.n_gmm,
            domain_disc=args_.domain_disc,
            hard_cluster=args_.hard_cluster,
            binary=args_.binary
        )
    else:
        global_learners_ensemble = \
            get_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary,
                phi_model=args.phi_model
            )

    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    print(args_.split, args_.hard_cluster)

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed,
            experiment = args_.experiment,
            method = args_.method,
            suffix = args_.suffix,
            split = args_.split,
            domain_disc=args_.domain_disc,
            em_step=args_.em_step
        )

    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    pre_action = 0
    mean_Is_pre = []
    rho = 0.3
    if_sufficient = False
    while current_round <= args_.n_rounds:


        if pre_action == 0:
            aggregator.mix(diverse=False)
        else:
            aggregator.mix(diverse=False)

        C = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]
        n_learner = aggregator.n_learners
        cluster_label_weights = [[0] * C for _ in range(n_learner)]
        cluster_weights = [0 for _ in range(n_learner)]
        global_flags = [[] for _ in range(n_learner)]
        if 'shakespeare' not in args_.experiment:
            with open('./logs/{}/sample-weight-{}-{}.txt'.format(args_.experiment, args_.method, args_.suffix), 'w') as f:
                for client_index, client in enumerate(clients):
                    for i in range(len(client.train_iterator.dataset.targets)):
                        if args_.method == 'FedSoft':
                            f.write('{},{},{}, {}\n'.format(client.data_type, client.train_iterator.dataset.targets[i], client.feature_types[i], aggregator.clusters_weights[client_index]))
                        else:
                            f.write('{},{},{}, {}\n'.format(client.data_type, client.train_iterator.dataset.targets[i], client.feature_types[i], client.samples_weights.T[i]))
                        
                        for j in range(len(cluster_label_weights)):
                            cluster_weights[j] += client.samples_weights[j][i]
                    f.write('\n')
        else:
            for client_index, client in enumerate(clients):
                for i in range(len(client.train_iterator.dataset.targets)):
                    for j in range(len(cluster_label_weights)):
                            cluster_weights[j] += client.samples_weights[j][i]

        with open('./logs/{}/mean-I-{}-{}-{}.txt'.format(args_.experiment, args_.method, args_.gamma, args_.suffix), 'a+') as f:
            mean_Is = torch.zeros((len(clients),))
            clusters = torch.zeros((len(clients),))
            client_types = torch.zeros((len(clients),))
            for i, client in enumerate(clients):
                mean_Is[i] = client.mean_I
                client_types[i] = client.data_type
                # clusters[i] = torch.nonzero(client.cluster==torch.max(client.cluster)).squeeze()
            f.write('{}'.format(mean_Is))
            f.write('\n')
        with open('./logs/{}/cluster-weights-{}-{}-{}.txt'.format(args_.experiment, args_.method, args_.gamma, args_.suffix), 'a+') as f:
            f.write('{}'.format(cluster_weights))
            f.write('\n')
        print(cluster_weights)
        # print(client_types)
        # print(clusters)



        # K = 0
        # for i in range(n_learner):
        #     if n_learner == 1:
        #         break
        #     if cluster_weights[i] <= sum(cluster_weights) * args_.gamma:
        #         # print(i)
        #         for client in clients:
        #             client.remove_learner(i - K)
        #         for client in test_clients:
        #             client.remove_learner(i - K)
        #         aggregator.remove_learner(i - K)
        #         K += 1
        #         cluster_label_weights.pop(i - K)
        #         if_sufficient = True
        

        for client in clients:
            client_labels_learner_weights = client.labels_learner_weights
            for j in range(len(cluster_label_weights)):
                for k in range(C):
                    cluster_label_weights[j][k] += client_labels_learner_weights[j][k]
        for j in range(len(cluster_label_weights)):
            for i in range(len(cluster_label_weights[j])):
                if cluster_label_weights[j][i] < 1e-8:
                    cluster_label_weights[j][i] = 1e-8
            cluster_label_weights[j] = [i / sum(cluster_label_weights[j]) for i in cluster_label_weights[j]]


        for client in clients:
            client.update_labels_weights(cluster_label_weights)
        # for client in test_clients:
        #     client.update_labels_weights(cluster_label_weights)

        for client in test_clients:
            print(client.mean_I, client.cluster, torch.nonzero(client.cluster==torch.max(client.cluster)).squeeze())

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round


    if "save_dir" in args_:
        save_dir = os.path.join(args_.save_dir)

        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    run_experiment(args)
