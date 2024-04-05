import sys

import argparse as ap
import pandas as pd
import numpy as np
from utils import *
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def main(args):
    argp = ap.ArgumentParser(description="extract features for dsrna prediction from randomly sampled region",
                             formatter_class=ap.ArgumentDefaultsHelpFormatter)
    argp.add_argument(
        "-i", "--input_file",
        help="input file of training and validation data",
        type=str,
        default="/data/dsRID_data.tsv"
    )

    argp.add_argument(
        "-p", "--pred_file",
        help="input file of prediction data",
        type=str,
        default="/data/dsRID_whole.tsv"
    )

    argp.add_argument(
        "-o", "--out_dir",
        help="output directory location",
        type=str,
        default="/data/"
    )

    argp.add_argument(
        "-m", "--model",
        help="model type",
        type=str,
        default="randomf"
    )

    args = argp.parse_args(args)

    modeltype = args.model

    model_dic = {
      "logireg" : LogisticRegression(random_state=0),
      "randomf" : RandomForestClassifier(random_state=0, bootstrap=True,
    criterion="gini", max_features=0.2, min_samples_leaf=4,
    min_samples_split=2, n_estimators=100),
      "svm" : LinearSVC(random_state=0, max_iter=2000)
    }

    model = model_dic[modeltype]

    all_frame = pd.read_csv(args.input_file,
      sep='\t')

    all_frame = all_frame.fillna(0)

    #all_frame['Alu'] = all_frame['dist'].apply(lambda x : 1 if x==0 else 0)
    # all_alt_frame = all_frame.loc[all_frame['label'] == 1].sample(35000)

    # all_null_frame = all_frame.loc[all_frame['label'] == 0].sample(35000)

    # train_alt_frame = all_alt_frame.iloc[:30000]

    # train_null_frame = all_null_frame.iloc[:30000]

    train_X, val_X, train_y, val_y = train_test_split(all_frame, all_frame['label'], test_size=0.33, random_state=42)
    # train_X = train_frame[['std_start','len_skip', 'dist']]

    # train_y = train_frame['label']

    # val_X = val_frame[['std_start','len_skip', 'dist']]

    # val_y = val_frame['label']

    cols = ~train_X.columns.isin(["mean_start", "mean_end",
     "coverage", "num_skip", "name",  "chr", "start", "end", "label"])

    train_cols = train_X.columns[cols]
    print("Using for training and fitting {}".format(' '.join(train_cols)))

    print(train_X.loc[:, cols])

    model = model.fit(train_X.loc[:, cols], train_y)

    #train_sc = model.score(train_X, train_y)

    #train_pred = model.predict_proba(train_X.loc[:, cols])


    val_pred = model.predict_proba(val_X.loc[:, cols])
    # Compute validation accuracy, precision, recall
    val_pred_labels = np.argmax(val_pred, axis=1)
    val_scores = {
        "accuracy" : metrics.accuracy_score(val_y, val_pred_labels),
        "precision" : metrics.precision_score(val_y, val_pred_labels),
        "recall" : metrics.recall_score(val_y, val_pred_labels),
        "roc_auc" : metrics.roc_auc_score(val_y, val_pred_labels)
    }
    pd.DataFrame(val_scores, index=[0]).to_csv(args.out_dir + "/val_scores_{}.tsv".format(modeltype), sep='\t', index=False)

    print(train_X.columns[cols])

    feat_imp = permutation_importance(model, val_X.loc[:, cols], val_y, scoring='r2')

    feat_imp_df = pd.DataFrame(data={'feat' : train_X.columns[cols], "mean_R2" : feat_imp.importances_mean,
                                      'std_R2' : feat_imp.importances_std})

    print(feat_imp_df)

    feat_imp_df.to_csv(args.out_dir + "/feat_importance_{}.tsv".format(modeltype), sep='\t', index=False)

    print(feat_imp.importances_mean, feat_imp.importances_std)

    scores = cross_validate(model, train_X.loc[:, cols], train_y, cv=5, scoring=['accuracy', 'precision', 'recall', 'roc_auc'])

    sc_frame = pd.DataFrame(scores)

    sc_frame.to_csv(args.out_dir + "/cv_scores_{}.tsv".format(modeltype),
    sep='\t', index=False)

    print(scores)

    train_pred = model.predict_proba(train_X.loc[:, cols])

    train_X['pred_0'] = train_pred[:, 0]

    train_X['pred_1'] = train_pred[:, 1]

    pred_frame = pd.read_csv(args.pred_file, sep='\t')

    pred_frame = pred_frame.fillna(0)

    # Make sure the columns are in the same order as the training data
    whole_pred = model.predict_proba(pred_frame.loc[:, train_cols])

    pred_frame['pred_0'] = whole_pred[:, 0]

    pred_frame['pred_1'] = whole_pred[:, 1]

    pred_frame.to_csv(args.out_dir + "/dsRID_prediction_{}.tsv".format(modeltype), sep='\t', index=False)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
