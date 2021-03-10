import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
all_envs[0]: non_crippled environment => used to train models of exp1
all_envs[i<6]: training environments => used to train models of exp2
all_envs[i>=6]: evaluation environments
"""
all_envs = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 0, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 0, 1, 1, 1],
                     [0.5, 1, 1, 1, 0, 0.3, 1, 1],
                     [0.8, 0.9, 0.6, 0.8, 0.5, 1, 0.7, 1],
                     [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1],
                     [1, 1, 1, 0, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 0, 1],
                     [1, 0.2, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 0.4, 1, 1, 1],
                     [1, 0.8, 0.1, 1, 0.5, 0.2, 1, 1],
                     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

iterables = [["normal", "meta", "online_adaptation"], ["non_crippled", "all"],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
index = pd.MultiIndex.from_product(iterables, names=["model_type", "trained_on", "env"])

rewards = [
    -6.052220763772775,  # normal, non_crippled, non_crippled
    -11.33884661082982,  # normal, non_crippled, training_0
    -6.426438334298114,  # normal, non_crippled, training_1
    -6.922354944732701,  # normal, non_crippled, training_2
    -9.080931834630373,  # normal, non_crippled, training_3
    -10.627915742366554,  # normal, non_crippled, training_4
    -89.80985766578884,  # normal, non_crippled, eval_0
    -7.000280094938234,  # normal, non_crippled, eval_1
    -45.30840069121141,  # normal, non_crippled, eval_2
    -6.60798206375808,  # normal, non_crippled, eval_3
    -10.386827524511702,  # normal, non_crippled, eval_4
    -22.139314536538585,  # normal, non_crippled, eval_5

    -7.2406269665778416,  # normal, all, non_crippled
    -15.995553285541689,  # normal, all, training_0
    -6.211190652536313,  # normal, all, training_1
    -6.698659396167691,  # normal, all, training_2
    -9.659393140055588,  # normal, all, training_3
    -10.742851273864092,  # normal, all, training_4
    -68.76171324396006,  # normal, all, eval_0
    -6.686968425065016,  # normal, all, eval_1
    -45.38330376096909,  # normal, all, eval_2
    -6.970001651497267,  # normal, all, eval_3
    -10.86810575535716,  # normal, all, eval_4
    -22.905402523719047,  # normal, all, eval_5

    -6.780300656955875,  # meta, non_crippled, non_crippled
    -7.127840142303028,  # meta, non_crippled, training_0
    -6.354007707817507,  # meta, non_crippled, training_1
    -7.052441990346658,  # meta, non_crippled, training_2
    -8.95732059775686,  # meta, non_crippled, training_3
    -10.03282017388594,  # meta, non_crippled, training_4
    -57.09393801645142,  # meta, non_crippled, eval_0
    -6.715347808804008,  # meta, non_crippled, eval_1
    -48.11385053093221,  # meta, non_crippled, eval_2
    -6.749148815526078,  # meta, non_crippled, eval_3
    -10.044744497915861,  # meta, non_crippled, eval_4
    -23.145782908625407,  # meta, non_crippled, eval_5

    -6.162745616430575,  # meta, all, non_crippled
    -6.535386135061344,  # meta, all, training_0
    -6.376819607538346,  # meta, all, training_1
    -6.89128019707804,  # meta, all, training_2
    -8.533103337965821,  # meta, all, training_3
    -10.893770709938957,  # meta, all, training_4
    -64.14196542822222,  # meta, all, eval_0
    -6.4944861013019795,  # meta, all, eval_1
    -51.09051028971922,  # meta, all, eval_2
    -6.40669101995659,  # meta, all, eval_3
    -9.403553355413734,  # meta, all, eval_4
    -22.356755848014494,  # meta, all, eval_5

    -6.412747365499675,  # online_adaptation, non_crippled, non_crippled
    -7.182915838295863,  # online_adaptation, non_crippled, training_0
    -6.187693671181198,  # online_adaptation, non_crippled, training_1
    -6.713665812907615,  # online_adaptation, non_crippled, training_2
    -8.418630998960037,  # online_adaptation, non_crippled, training_3
    -9.631350835422202,  # online_adaptation, non_crippled, training_4
    -57.148050647268676,  # online_adaptation, non_crippled, eval_0
    -6.681012142167364,  # online_adaptation, non_crippled, eval_1
    -48.77529859290009,  # online_adaptation, non_crippled, eval_2
    -6.826560706510262,  # online_adaptation, non_crippled, eval_3
    -10.02255968472495,  # online_adaptation, non_crippled, eval_4
    -22.37246305468445,  # online_adaptation, non_crippled, eval_5

    -6.257941418879788,  # online_adaptation, all, non_crippled
    -6.651656928829949,  # online_adaptation, all, training_0
    -6.566841730361049,  # online_adaptation, all, training_1
    -6.54218110996944,  # online_adaptation, all, training_2
    -8.370704966383727,  # online_adaptation, all, training_3
    -10.822433429526509,  # online_adaptation, all, training_4
    -57.99585889227632,  # online_adaptation, all, eval_0
    -6.753815841021042,  # online_adaptation, all, eval_1
    -51.58070789688873,  # online_adaptation, all, eval_2
    -6.468923119288346,  # online_adaptation, all, eval_3
    -10.081977904586246,  # online_adaptation, all, eval_4
    -22.773969815987485,  # online_adaptation, all, eval_5
]

average_reward = pd.Series(rewards, index=index)

# plot performance on non_crippled task
average_reward[:, :, 0].unstack().plot(kind="bar")
plt.title("Average Reward for non-crippled Environment")
plt.show()

for i in range(1, 12):
    # plot performance on non_crippled task
    average_reward[:, :, i].unstack().plot(kind="bar")
    plt.title(f"Average Reward for {all_envs[i]}")
    plt.show()

# plot performance of normal
test = average_reward.swaplevel(i=1, j=2, )
test["normal", :, :].unstack().plot(kind="bar", )
plt.title("Average Reward of 'normal' algorithm on all Environments")
plt.show()

# plot performance of meta
test = average_reward.swaplevel(i=1, j=2, )
test["meta", :, :].unstack().plot(kind="bar", )
plt.title("Average Reward of meta algorithm on all Environments")
plt.show()

# plot performance of online_adaptation
test = average_reward.swaplevel(i=1, j=2, )
test["online_adaptation", :, :].unstack().plot(kind="bar", )
plt.title("Average Reward of 'normal' algorithm with online adaptation on all Environments")
plt.show()

# plot performance of all models trained on non crippled data only
test = average_reward.swaplevel(i=0, j=2, )
test[:, "non_crippled", :].unstack().plot(kind="bar")
plt.title("Average Reward of models trained with non-crippled data only")
plt.show()

# plot performance of all models trained on all data of training set
test = average_reward.swaplevel(i=0, j=2, )
test[:, "all", :].unstack().plot(kind="bar")
plt.title("Average Reward of models trained with all data")
plt.show()
