import torch
import torch.nn as nn

from plan import Operator

operators = ["Projection", "Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableReader",
             "IndexReader", "IndexLookUp"]


# There are many ways to extract features from plan:
# 1. The simplest way is to extract features from each node and sum them up. For example, we can get
#      a  the number of nodes;
#      a. the number of occurrences of each operator;
#      b. the sum of estRows for each operator.
#    However we lose the tree structure after extracting features.
# 2. The second way is to extract features from each node and concatenate them in the DFS traversal order.
#                  HashJoin_1
#                  /          \
#              IndexJoin_2   TableScan_6
#              /         \
#          IndexScan_3   IndexScan_4
#    For example, we can concatenate the node features of the above plan as follows:
#    [Feat(HashJoin_1)], [Feat(IndexJoin_2)], [Feat(IndexScan_3)], [Padding], [Feat(IndexScan_4)], [Padding], [Padding], [Feat(TableScan_6)], [Padding], [Padding]
#    Notice1: When we traverse all the children in DFS, we insert [Padding] as the end of the children. In this way, we
#    have an one-on-one mapping between the plan tree and the DFS order sequence.
#    Notice2: Since the different plans have the different number of nodes, we need padding to make the lengths of the
#    features of different plans equal.
class PlanFeatureCollector:
    def __init__(self):
        # YOUR CODE HERE: define variables to collect features from plans
        self.nodes = 1
        self.operators = [0,0,0,0,0,0,0,0,0,0]
        self.estRows = [0,0,0,0,0,0,0,0,0,0]

    def add_operator(self, op: Operator):
        # YOUR CODE HERE: extract features from op
        self.nodes += 1
        if op.is_projection():
            self.operators[0] += 1
            self.estRows[0] += op.est_row_counts()
        elif op.is_selection():
            self.operators[1] += 1
            self.estRows[1] += op.est_row_counts()
        elif op.is_sort():
            self.operators[2] += 1
            self.estRows[2] += op.est_row_counts()
        elif op.is_hash_agg():
            self.operators[3] += 1
            self.estRows[3] += op.est_row_counts()
        elif op.is_hash_join():
            self.operators[4] += 1
            self.estRows[4] += op.est_row_counts()
        elif op.is_table_scan():
            self.operators[5] += 1
            self.estRows[5] += op.est_row_counts()
        elif op.is_index_scan():
            self.operators[6] += 1
            self.estRows[6] += op.est_row_counts()
        elif op.is_table_reader():
            self.operators[7] += 1
            self.estRows[7] += op.est_row_counts()
        elif op.is_index_reader():
            self.operators[8] += 1
            self.estRows[8] += op.est_row_counts()
        elif op.is_index_lookup():
            self.operators[9] += 1
            self.estRows[9] += op.est_row_counts()
        else:        
            print(operator.id)
            assert (1 == 2)  # unknown operator


    def walk_operator_tree(self, op: Operator):
        self.add_operator(op)
        for child in op.children:
            self.walk_operator_tree(child)
        # YOUR CODE HERE: process and return the features
        return [self.nodes, operator in self.operators, estRow in self.estRows]


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, plans, max_operator_num):
        super().__init__()
        self.data = []
        for plan in plans:
            collector = PlanFeatureCollector()
            vec = collector.walk_operator_tree(plan.root)
            # YOUR CODE HERE: maybe you need padding the features if you choose the second way to extract the features.
            features = torch.Tensor(vec)
            exec_time = torch.Tensor([plan.exec_time_in_ms()])
            self.data.append((features, exec_time))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Define your model for cost estimation
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # YOUR CODE HERE
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # YOUR CODE HERE
        pass

    def init_weights(self):
        # YOUR CODE HERE
        pass


def count_operator_num(op: Operator):
    num = 2  # one for the node and another for the end of children
    for child in op.children:
        num += count_operator_num(child)
    return num


def estimate_learning(train_plans, test_plans):
    max_operator_num = 0
    for plan in train_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    for plan in test_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    print(f"max_operator_num:{max_operator_num}")

    train_dataset = PlanDataset(train_plans, max_operator_num)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=1)

    model = YourModel()
    model.init_weights()

    def loss_fn(est_time, act_time):
        # YOUR CODE HERE: define loss function
        pass

    # YOUR CODE HERE: complete training loop
    num_epoch = 20
    for epoch in range(num_epoch):
        print(f"epoch {epoch} start")
        for i, data in enumerate(train_loader):
            pass

    train_est_times, train_act_times = [], []
    for i, data in enumerate(train_loader):
        # YOUR CODE HERE: evaluate on train data
        pass

    test_dataset = PlanDataset(test_plans, max_operator_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    test_est_times, test_act_times = [], []
    for i, data in enumerate(test_loader):
        # YOUR CODE HERE: evaluate on test data
        pass

    return train_est_times, train_act_times, test_est_times, test_act_times
