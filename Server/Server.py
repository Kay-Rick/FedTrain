import torch
import os
import numpy as np
import copy


class Server:
    def __init__(
        self,
        device,
        dataset,
        algorithm,
        model,
        batch_size,
        learning_rate,
        num_glob_iters,
        local_epochs,
        optimizer,
        num_users,
        times,
    ):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.algorithm = algorithm
        (
            self.rs_train_acc,
            self.rs_glob_loss,
            self.rs_glob_acc,
            self.rs_train_acc_per,
            self.rs_train_loss_per,
            self.rs_glob_acc_per,
        ) = ([], [], [], [], [], [])
        self.rs_train_loss = []
        self.times = times
        self.rs_glob_Jacobian_loss = []

    def aggregate_grads(self):
        assert self.users is not None and len(self.users) > 0
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert self.users is not None and len(self.users) > 0
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(
            self.model.parameters(), user.get_parameters()
        ):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert self.users is not None and len(self.users) > 0
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("TrainedModels", self.dataset)
        a = ""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if self.n_attackers == 0:
            self.attacker_type = "Noattack"
        if self.num_goodUsers > 0:
            a = "_" + "WithModelJacobian"

        torch.save(
            self.model,
            os.path.join(
                model_path, self.attacker_type + "_" + self.algorithm + a + ".pt"
            ),
        )

    def load_model(self, model_path):
        assert os.path.exists(model_path)
        print("model loaded!")
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users):
        """selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        """
        if num_users == len(self.users):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        # np.random.seed(round)
        # , p=pk)
        return np.random.choice(self.users, num_users, replace=False)

    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = (
            alg
            + "_"
            + str(self.num_users)
            + "_"
            + self.attacker_type
            + "_"
            + str(self.n_attackers)
            + "_"
            + "GoodUsers"
            + "_"
            + str(self.num_goodUsers)
        )
        alg = alg + "_" + str(self.times)
        alg_acc = alg + "_acc.npy"
        alg_loss = alg + "_loss.npy"
        np.save(os.path.join("result", alg_acc), self.rs_glob_acc)
        print("-----------" + alg_acc + "-----------")
        print(np.load(os.path.join("result", alg_acc)))
        np.save(os.path.join("result", alg_loss), self.rs_train_loss)
        print("-----------" + alg_loss + "-----------")
        print(np.load(os.path.join("result", alg_loss)))

    def save_jacobian(self):
        alg = self.dataset + "_" + self.algorithm
        alg = (
            alg
            + "_"
            + str(self.num_users)
            + "_"
            + self.attacker_type
            + "_"
            + str(self.n_attackers)
            + "_"
            + "GoodUsers"
            + "_"
            + str(self.num_goodUsers)
        )
        alg = alg + "_" + str(self.times)
        alg_loss = alg + "_Jacobianloss"
        np.save(os.path.join("CurveResult", alg_loss), self.rs_glob_Jacobian_loss)

    def test(self):
        """tests self.latest_model on given clients"""
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct

    def testJacobian(self):
        Jloss = 0
        for c in self.users:
            jloss = c.testJacobian()
            Jloss = Jloss + jloss
        return Jloss

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum(
            [x * y for (x, y) in zip(stats_train[1], stats_train[3])]
        ).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        # print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ", train_loss)

    def evaluateJacobian(self):
        JacobianLoss = self.testJacobian()
        JacobianLoss = JacobianLoss / self.num_users
        self.rs_glob_Jacobian_loss.append(JacobianLoss)
        print("Average Global Jacobian Regularization: ", JacobianLoss)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum(
            [x * y for (x, y) in zip(stats_train[1], stats_train[3])]
        ).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        # print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ", train_loss)
