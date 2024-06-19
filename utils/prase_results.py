

# file = './logs/cifar10-c/results-conceptEM-06-lr.txt'
# file = './logs/tiny-imagenet-c/results-conceptEM_tune-06-lr-12.txt'
# file = './logs/tiny-imagenet-c/results-conceptEM-06-lr-123.txt'
# file = './logs/cifar10-c/FedEM/results-conceptem-tune-resnet18.txt'
# file = './logs/airline-1/results-FedAvg-06-lr-123.txt'
# file = './logs/cifar10-c-2swap/results-FedAvg-06-lr.txt'
# file = './logs/cifar10-c-noisy-02/results-FedAvg-06-lr-123.txt'
# file = './logs/cifar10-c-noisy-type2-02/results-conceptEM_SW-06-lr-04-split.txt'
# file = './logs/cifar10-c-noisy-type2-04/results-conceptEM_SW-06-lr-split.txt'
# file = './logs/cifar10-c-4swap/results-conceptEM_SW-06-lr-split.txt'
# file = './logs/cifar100-c-4swap/results-conceptEM-06-lr-123.txt'
# file = './logs/cifar10-c-concept-only/results-FedAvg-06-lr.txt'
# file = './logs/cifar10-c-feature-only/results-conceptEM-06-lr.txt'
# file = './logs/cifar10-c-label-only/results-conceptEM-06-lr.txt'
# file = './logs/cifar10-c-concept-label/results-conceptEM-06-lr.txt'
# file = './logs/cifar10-c-concept-feature/results-conceptEM-06-lr.txt'
# file = './logs/cifar10-c-concept-feature-label/results-FedAvg-06-lr-decorr.txt'
# file = './logs/tiny-imagenet-c/results-conceptEM-06-lr-single-1.txt'
# file = './logs/cifar100-c-2swap/results-FedAvg-06-lr.txt'
# file = './logs/cifar100-c-4swap/results-stoCFL-03-lr-05-015-1.txt'
# file = './logs/cifar100-c-2swap/results-conceptEM_SW-03-lr-adapt-proto-mean-04-1.txt'
# file = './logs/cifar10-c-concept-feature-label/results-conceptEM-06-lr-5-model.txt'
# file = './logs/cifar10-c-4swap/results-FeSEM-03-lr-03-split-1.txt'
# count_file = './logs/cifar10-c-4swap/cluster-weights-FeSEM-0.05-03-lr-03-split-1.txt'
# file = './logs/cifar100-c-2swap/results-FedEM_SW-03-lr-005-adapt-split-1.txt'
# file = './logs/cifar100-c-2swap/results-ICFL-03-lr-4-098-1.txt'
file = './logs/tiny-imagenet-c-2swap/results-FedEM_SW-03-lr-005-0-adapt-split-1.txt'
# count_file = './logs/cifar100-c-4swap/cluster-weights-conceptEM_SW-0.0-03-lr-adapt-proto-03-1234.txt'
# count_file = './logs/tiny-imagenet-c-2swap/cluster-weights-conceptEM_SW-0.05-03-lr-03-04-adapt-split-12.txt'
file = './logs/cifar10-c-2swap/results-conceptEM_SW-03-lr-03-resnet-split-01-04-1.txt'

# max_count = 0
# final_count = 0
# with open(count_file, 'r') as f:
#     lines = f.readlines()
#     lines = [x.strip() for x in lines]
#     lines = [x.split(', ') for x in lines]
#     counts = [len(x) for x in lines]
#     max_count = max(counts)
#     final_count = counts[-1]

# print('final_count:' + str(final_count))
# print('max_count:' + str(max_count))


with open(file, 'r') as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    lines = [x.split(', ') for x in lines]
    lines = [[float(y) for y in x] for x in lines]

train_losses = []
train_acces = []
local_losses = []
local_acces = []
global_losses = []
global_acces = []

i = 0
for line in lines:
    # if i <= 410:
    #     i += 1
    #     continue
    if i % 2 == 0:
        train_losses.append(line[0])
        train_acces.append(line[1])
        local_losses.append(line[2])
        local_acces.append(line[3])
    elif i % 2 == 1:
        global_losses.append(float(line[2]))
        global_acces.append(float(line[3]))
    i += 1

print(i)

# round = train_acces.index(max(train_acces[:50]))
# round = global_acces.index(max(global_acces[:200]))
round = local_acces.index(max(local_acces))

print('local: {}'.format(local_acces[round]))

round = global_acces.index(max(global_acces[:200]))

print('global: {}'.format(global_acces[round]))

