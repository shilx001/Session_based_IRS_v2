import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from TreeActorCritic import *
from FeatureExtractor import *
import datetime
import matplotlib.pyplot as plt
from funk_svd import SVD

model_name = 'convLSTM'
np.random.seed(1)
#data = pd.read_csv('ratings.csv',header=0,names=['u_id', 'i_id', 'rating', 'timestep'])
data = pd.read_table('ratings.dat', sep='::', names=['u_id', 'i_id', 'rating', 'timestep'])
user_idx = data['u_id'].unique()  # id for all the user
np.random.shuffle(user_idx)
train_id = user_idx[:int(len(user_idx) * 0.8)]
test_id = user_idx[int(len(user_idx) * 0.8):]

# Count the movies
movie_id = []
for idx1 in user_idx:  # 针对train_id中的每个用户
    user_record = data[data['u_id'] == idx1]
    for idx2, row in user_record.iterrows():
        if row['i_id'] in movie_id:
            idx = movie_id.index(row['i_id'])  # 找到该位置
        else:
            # 否则新加入movie_id
            movie_id.append(row['i_id'])

# Build user rating matrix
rating_mat = np.zeros([len(train_id), len(movie_id)])
movie_id = np.array(movie_id)
for idx in train_id:  # 针对每个train数据
    record = data[data['u_id'] == idx]  # record有多个数据，所以row_index也有多个
    for _, row in record.iterrows():  # 针对每个用户的每条评分
        r = np.where(train_id == idx)
        c = np.where(row['i_id'] == movie_id)
        rating_mat[r, c] = row['rating']

# Funk SVD for item representation
train = data[data['u_id'].isin(train_id)]
test = data[data['u_id'].isin(test_id)]
svd = SVD(learning_rate=1e-3, regularization=0.005, n_epochs=100, n_factors=128, min_rating=0, max_rating=5)
svd.fit(X=data, X_val=test, early_stopping=True, shuffle=False)
item_matrix = svd.qi


def get_feature(input_id):
    # 根据输入的movie_id得出相应的feature
    movie_index = np.where(movie_id == input_id)
    return item_matrix[movie_index]


def action_mapping(input_id):
    '''
    convert input movie id to index
    :param input_id: movie id
    :return: index of movie.
    '''
    return np.where(movie_id == input_id)


max_seq_length = 32
state_dim = item_matrix.shape[1] + 1
hidden_size = 64

feature_extractor = FeatureExtractor(state_dim=state_dim, max_seq_length=max_seq_length, hidden_size=hidden_size,
                                     learning_rate=1e-3)
# Pre-training process to learn state features.
print('Begin pre-train.')
pre_train_step = 0
loss_list = []
for id1 in train_id[:100]:
    user_record = data[data['u_id'] == id1]
    state = []
    rating = []
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['i_id'])
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        rating.append(row['rating'])
        state.append(current_state)
    state_list = []
    length_list = []
    rating_list = []
    for i in range(1, len(state)):
        current_state = state[:i]
        current_state_length = i
        temp_state = feature_extractor.state_padding(current_state, current_state_length)
        length_list.append(current_state_length)
        state_list.append(temp_state)
        rating_list.append(rating[i])
    loss = feature_extractor.train(state_list, length_list, rating_list)
    loss_list.append(loss)
    pre_train_step += 1
    print('Pretrain step: ', pre_train_step, ' MSE:', loss)

plt.figure()
plt.plot(loss_list)
plt.title('Training loss')
plt.savefig('Train loss_' + model_name)

'''
print('Begin test for feature extraction.')
loss_list = []
for id1 in test_id:
    user_record = data[data['userId'] == id1]
    state = []
    rating = []
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['movieId'])
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        rating.append(row['rating'])
        state.append(current_state)
    state_list = []
    length_list = []
    rating_list = []
    for i in range(1, len(state)):
        current_state = state[:i]
        current_state_length = i
        temp_state = feature_extractor.state_padding(current_state, current_state_length)
        length_list.append(current_state_length)
        state_list.append(temp_state)
        rating_list.append(rating[i])
    loss = feature_extractor.get_loss(state_list, length_list, rating_list)
    loss_list.append(loss)
    pre_train_step += 1
    print('test loss: ', ' MSE:', loss)
'''

plt.figure()
plt.plot(loss_list)
plt.title('Test loss')
plt.savefig('Test loss_' + model_name)

agent = TreeActorCritic(state_dim=hidden_size, layer=3, branch=16, learning_rate=1e-4, discount_factor=0.99)


def normalize(rating):
    max_rating = 5
    min_rating = 0
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)


print('Begin training the tree actor-critic.')
train_step = 0
loss_list = []
for id1 in train_id:
    user_record = data[data['u_id'] == id1]
    state = []
    rating = []
    action = []
    next_state = []
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['i_id'])
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        rating.append(row['rating'])
        state.append(current_state)
        action.append(action_mapping(row['i_id']))
    state_list = []
    action_list = []
    reward_list = []
    next_state_list = []
    for i in range(2, len(state)):
        current_state = state[:i - 1]
        current_state_length = i - 1
        next_state = state[:i]
        next_state_length = i
        temp_state = feature_extractor.state_padding(current_state, current_state_length)
        temp_next_state = feature_extractor.state_padding(next_state, next_state_length)
        state_list.append(feature_extractor.get_feature(temp_state, [current_state_length]).flatten())
        next_state_list.append(feature_extractor.get_feature(temp_next_state, [next_state_length]).flatten())
        action_list.append(action[i])
        reward_list.append(normalize(rating[i]))  # normalization of the ratings to 0,1
    loss = agent.train(state_list, action_list, reward_list, next_state_list)
    loss_list.append(loss)
    train_step += 1
    print('Step ', train_step, 'Loss: ', loss)

plt.figure()
plt.plot(loss_list[0])
plt.title('Actor loss')
plt.savefig('actor loss')

plt.figure()
plt.plot(loss_list[1])
plt.title('Critic loss')
plt.savefig('critic loss')

print('Begin Test')
N = 30
test_count = 0
result = []


print('Begin Test')
N = 30
test_count = 0
result = []


def evaluate(recommend_id, item_id, rating, top_N):
    '''
    evalute the recommend result for each user.
    :param recommend_id: the recommend_result for each item, a list that contains the results for each item.
    :param item_id: item id.
    :param rating: user's rating on item.
    :param top_N: N, a real number of N for evaluation.
    :return: reward@N, recall@N, MRR@N
    '''
    session_length = len(recommend_id)
    relevant = 0
    recommend_relevant = 0
    output_reward = 0
    mrr = 0
    for ti in range(session_length):
        current_recommend_id = list(recommend_id[ti])[:top_N]
        current_item = item_id[ti]
        current_rating = rating[ti]
        if current_rating > 3.5:
            relevant += 1
            if current_item in current_recommend_id:
                recommend_relevant += 1
        if current_item in current_recommend_id:
            output_reward += normalize(current_rating)
            rank = current_recommend_id.index(current_item)
            mrr += 1.0 / (rank + 1)
    recall = recommend_relevant / relevant if relevant is not 0 else 0
    precision = recommend_relevant / session_length
    return output_reward / session_length, precision, recall, mrr / session_length


for id1 in test_id:
    user_record = data[data['u_id'] == id1]
    all_state = []
    count = 0
    all_recommend = []
    all_item = []
    all_rating = []
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['i_id'])
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        all_state.append(current_state)
        if len(all_state) > 1:
            count += 1
            temp_state = all_state[:-1]
            temp_state_length = len(temp_state)
            temp_state = feature_extractor.state_padding(temp_state, temp_state_length)
            state_feature = feature_extractor.get_feature(temp_state, [temp_state_length])
            output_action = agent.get_action_prob(state_feature).flatten()
            output_action = output_action[:len(movie_id)]
            recommend_idx = np.argsort(-output_action)[:30]
            recommend_movie = movie_id[recommend_idx]
            all_recommend.append(recommend_movie)
            all_item.append(row['i_id'])
            all_rating.append(row['rating'])
    reward_10, precision_10, recall_10, mkk_10 = evaluate(all_recommend, all_item, all_rating, 10)
    reward_30, precision_30, recall_30, mkk_30 = evaluate(all_recommend, all_item, all_rating, 30)
    test_count += 1
    print('Test user #', test_count, '/', len(test_id))
    print('Reward@10: %.4f, Precision@10: %.4f, Recall@10: %.4f, MRR@10: %4f'
          % (reward_10, precision_10, recall_10, mkk_10))
    print('Reward@30: %.4f, Precision@30: %.4f, Recall@30: %.4f, MRR@30: %4f'
          % (reward_30, precision_30, recall_30, mkk_30))
    result.append([reward_10, precision_10, recall_10, mkk_10, reward_30, precision_30, recall_30, mkk_30])

pickle.dump(result, open('tpgr_' + model_name, mode='wb'))
print('Result:')
print(np.mean(np.array(result).reshape([-1, 8]), axis=0))
