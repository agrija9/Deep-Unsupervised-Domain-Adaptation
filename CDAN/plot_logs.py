import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import numpy

def plot_loss_acc(root_dir,source,target,method):
  path_adapt_log = [root_dir+"/training_s_statistic.pkl",
                    root_dir+"/testing_s_statistic.pkl",
                    root_dir+"/testing_t_statistic.pkl"]

  path_no_adapt_log = [root_dir+"/no_adaptation_training_s_statistic.pkl",
                       root_dir+"/no_adaptation_testing_s_statistic.pkl",
                       root_dir+"/no_adaptation_testing_t_statistic.pkl"]

  adapt_training_dict = pickle.load(open(path_adapt_log[0], 'rb'))
  adapt_testing_source_dict = pickle.load(open(path_adapt_log[1], 'rb'))
  adapt_testing_target_dict = pickle.load(open(path_adapt_log[2], 'rb'))

  no_adapt_training_dict = pickle.load(open(path_no_adapt_log[0], 'rb'))
  no_adapt_testing_source_dict = pickle.load(open(path_no_adapt_log[1], 'rb'))
  no_adapt_testing_target_dict = pickle.load(open(path_no_adapt_log[2], 'rb'))
  if method == "CDAN":
        adaptation = {
            "classification_loss": [],
            "cdan_loss": [],
            "source_accuracy": [],
            "target_accuracy": []
        }
  elif method == "CDAN+E":
        adaptation = {
            "classification_loss": [],
            "cdan_e_loss": [],
            "source_accuracy": [],
            "target_accuracy": []
        }

  no_adaptation = {
      "source_accuracy": [],
      "target_accuracy": []
  }

  # get average coral and classification loss for steps in each epoch
  # for epoch_idx in range(len(adapt_training_dict)-1): # epoch
  for epoch_idx in range(100): # epoch

    transfer_loss = 0
    class_loss = 0

    for step_idx in range(len(adapt_training_dict[epoch_idx])):
      if method == "CDAN":
          transfer_loss += adapt_training_dict[epoch_idx][step_idx]["cdan_loss"]
      elif method == "CDAN+E":
          transfer_loss += adapt_training_dict[epoch_idx][step_idx]["cdan_e_loss"]
      class_loss += adapt_training_dict[epoch_idx][step_idx]["classification_loss"]

    # store average losses and accuracies in adaptation dictionary
    adaptation["classification_loss"].append(class_loss/len(adapt_training_dict[epoch_idx]))
    if method == "CDAN":
        adaptation["cdan_loss"].append(transfer_loss/len(adapt_training_dict[epoch_idx]))
    elif method == "CDAN+E":
        adaptation["cdan_e_loss"].append(transfer_loss/len(adapt_training_dict[epoch_idx]))
    adaptation["source_accuracy"].append(adapt_testing_source_dict[epoch_idx]["accuracy %"])
    adaptation["target_accuracy"].append(adapt_testing_target_dict[epoch_idx]["accuracy %"])
    # print(epoch_idx,len(no_adapt_testing_source_dict))
    # # store accuracies in no-adaptation dictionary
    no_adaptation["source_accuracy"].append(no_adapt_testing_source_dict[epoch_idx]["accuracy %"])
    no_adaptation["target_accuracy"].append(no_adapt_testing_target_dict[epoch_idx]["accuracy %"])
  print(np.mean(adaptation['target_accuracy']),np.std(adaptation['target_accuracy']))
  # plot accuracies for test data in source and target domains
  fig=plt.figure(figsize=(8, 6), dpi=100)
  fig.show()
  plt.title(source+" to "+target)

  plt.xlabel("epochs", fontsize=15)
  plt.ylabel("classification accuracy (%)", fontsize=15)

  plt.plot(adaptation['target_accuracy'], label="test acc. w/ transfer loss", marker='*', markersize=8)
  plt.plot(no_adaptation['target_accuracy'], label="test acc. w/o transfer loss", marker='.', markersize=8)

  plt.plot(adaptation['source_accuracy'], label="training acc. w/ transfer loss", marker='^', markersize=8)
  plt.plot(no_adaptation['source_accuracy'], label="training acc. w/o transfer loss", marker='+', markersize=8)

  plt.legend(loc="best")
  plt.grid()
  plt.show()
  fig.savefig(root_dir+"/"+source+"_to_"+target+"_test_train_accuracies.png")

  # plot accuracies for test data in source and target domains
  fig=plt.figure(figsize=(8, 6), dpi=100)
  fig.show()

  plt.xlabel("epochs", fontsize=15)
  plt.ylabel("Loss", fontsize=15)
  plt.title(source+" to "+target)
  plt.plot(adaptation["classification_loss"], label="classification_loss", marker='*', markersize=8)
  if method == "CDAN":
      plt.plot(adaptation["cdan_loss"], label="transfer_loss", marker='.', markersize=8)
  elif method == "CDAN+E":
      plt.plot(adaptation["cdan_e_loss"], label="transfer_loss", marker='.', markersize=8)

  plt.legend(loc="best")
  plt.grid()
  plt.show()
  fig.savefig(root_dir+"/"+source+"_to_"+target+"_train_losses.png")

def main():
    parser = argparse.ArgumentParser(description="plots graphs for CDAN and CDAN+E transfer loss ")
    parser.add_argument("--root_dir",default="CDAN/amazon_to_dslr", type=str)
    parser.add_argument("--source", default="amazon", type=str,
                        help="source data")

    parser.add_argument("--target", default="dslr", type=str,
                        help="target data")

    parser.add_argument("--method", default="CDAN", type=str)

    args = parser.parse_args()

    plot_loss_acc(root_dir= args.root_dir, source=args.source, target=args.target, method=args.method)


if __name__ == "__main__":
    main()
