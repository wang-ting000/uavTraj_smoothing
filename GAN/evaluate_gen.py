import numpy as np
from scipy import stats


###############################################t-检验
# 生成样本数据（假设生成数据和真实数据已经准备好）
qs = np.load('../CKM/qs.npy')[0:1000]

qs /= np.max(qs, axis=0)


gs = np.load('../CKM/channel_gains.npy')[0:1000]
gs /= np.max(gs, axis=0)
real_data = np.hstack((qs, gs))

generated_data = np.load('new_data.npy')

print(generated_data)



########################################################KL散度
# import numpy as np
# from scipy.special import rel_entr
#
# # 假设生成数据和真实数据的概率分布
# generated_prob = np.random.dirichlet(np.ones(10), size=1)[0]  # 示例生成数据概率分布
# real_prob = np.random.dirichlet(np.ones(10), size=1)[0]  # 示例真实数据概率分布
#
# # 计算KL散度
# kl_divergence = np.sum(rel_entr(real_prob, generated_prob))
#
# print(f"KL Divergence: {kl_divergence}")
#
# ###################################################JS散度
# import numpy as np
# from scipy.spatial.distance import jensenshannon
#
# # 假设生成数据和真实数据的概率分布
# generated_prob = np.random.dirichlet(np.ones(10), size=1)[0]  # 示例生成数据概率分布
# real_prob = np.random.dirichlet(np.ones(10), size=1)[0]  # 示例真实数据概率分布
#
# # 计算JS散度
# js_divergence = jensenshannon(generated_prob, real_prob, base=2)
#
# print(f"JS Divergence: {js_divergence}")

####################################################K-S检验





