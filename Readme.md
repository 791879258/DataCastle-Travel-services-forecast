
###Readme
@(DataCastle)


----------


**DataCastle Travel services forecast**精品旅行服务成单预测比赛——[比赛官网](http://www.dcjingsai.com/common/cmpt/%E7%B2%BE%E5%93%81%E6%97%85%E8%A1%8C%E6%9C%8D%E5%8A%A1%E6%88%90%E5%8D%95%E9%A2%84%E6%B5%8B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)

----------
比赛简介：
- 任务：在这个比赛中，我们提供了5万多名用户在旅游app中的浏览行为记录，其中有些用户在浏览之后完成了订单，且享受了精品旅游服务，而有些用户则没有下单。
参赛者需要分析用户的个人信息和浏览行为，从而预测用户是否会在短期内购买精品旅游服务；
- 数据集：用户个人信息数据，用户行为数据，用户历史订单数据，评论数据，待预测订单数据。

----------

原始数据集放在origin_data文件夹；
代码放在DC_Code中；
- data是程序操作的数据集，即为train和test文件，和生成的一些中间文件；
- model 表示训练的模型
- preprocess为数据的预处理文件夹，里面有数据预处理(preprocess.py)和构造特征的代码(construct_feature.py和construct_feature_from_jingsaiquan.py)，及生成的特征文件(在feature_file中)，以及生成的最终用于训练和预测的数据集(xxxdata_output文件夹)；
- result文件夹中为生成的结果文件，可以直接提交；


----------
目前的特征包括自己构建的特征，对应特征文件生成代码为`construct_feature.py`：
- 每个user各行为类型的次数(9种行为，9列特征)
- 年龄和省份，通过OneHot编码操作；
- 用户是否购买过精品旅游服务ever_bought_orderType1；
- 用户购买过服务的次数（包括普通服务和精品旅游）user_purchase_orderType01_times；
- 用户评分记录中是否存在>=3的记录if_exist_rate_largerthan_2；
- 用户购买了普通服务且评分>=3的记录 user_orderType0_rate_largerthan_2；
- 用户点击率特征：
user_click_num表示用户点击次数
user_click_rate用户点击率
user_purchase_num用户购买服务的次数
 user_purchase_rate用户购买率
user_open_app_num用户打开app的次数
 user_click_but_no_purchase用户点击但不消费次数；
- 用户在提交最后一次订单后继续浏览产品的次数actionType234_after_submit_last_order；
- 用户评论数据中tags和keywords数量，tags_num and keywords_num；
- TypeX_num_by_day这9个特征和days，表示action记录中用户平均每天的各Type数量以及天数；
- TypeX_last1_day这9个特征，表示最后一天TypeX的次数；
- TypeX_last3_day这9个特征，表示最后三天TypeX的次数；
- order_day_mean下单天数跨度的均值，day_after_lastorder最后一次订单后距最后一次actionTime天数跨度；
- to_Xorder_timetamp距离某类订单的时间戳信息;
- order_time_interval距离某类订单时间间隔信息;
- action_TypeX_interval_time_info_XXX action中各Type的时间间隔均值等信息

和[竞赛圈开源的特征。](http://www.dcjingsai.com/common/bbs/topicDetails.html?tid=637)对应的特征生成代码为`construct_feature_from_jingsaiquan.py`


----------

黄包车这个比赛可以说是自己完全独立做的一个比赛了，在快结束前和队友进行了模型的融合。
比赛中的一些想法记录见**黄包车比赛想法记录.md**。
生成的特征文件csv在**/preprocess/feature_file/**
生成的训练集和测试集文件csv在**/preprocess/traindata_output和testdata_output文件夹中**
模型的训练和对测试集的预测在**/model/xgb_CV.py**
尝试把本问题作为推荐问题，使用mahout对各个user的行为和购买数据进行相似性度量，列出某个user最相似的topK个userid作为特征，但是效果并没有提升，这部分代码见`/mahout_DT`

----------
上述特征文件的结果为96339，和队友的概率文件融合达到96630.