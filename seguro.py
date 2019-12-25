from pyspark.ml import Pipeline

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import Imputer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import PCA
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import *

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

# Schema definition
schema=StructType([
    StructField("id", FloatType(), True),
    StructField("target", FloatType(), True),
    StructField("ps_ind_01", FloatType(), True),
    StructField("ps_ind_02_cat", FloatType(), True),
    StructField("ps_ind_03", FloatType(), True),
    StructField("ps_ind_04_cat", FloatType(), True),
    StructField("ps_ind_05_cat", FloatType(), True),
    StructField("ps_ind_06_bin", FloatType(), True),
    StructField("ps_ind_07_bin", FloatType(), True),
    StructField("ps_ind_08_bin", FloatType(), True),
    StructField("ps_ind_09_bin", FloatType(), True),
    StructField("ps_ind_10_bin", FloatType(), True),
    StructField("ps_ind_11_bin", FloatType(), True),
    StructField("ps_ind_12_bin", FloatType(), True),
    StructField("ps_ind_13_bin", FloatType(), True),
    StructField("ps_ind_16_bin", FloatType(), True),
    StructField("ps_ind_17_bin", FloatType(), True),
    StructField("ps_ind_18_bin", FloatType(), True),
    StructField("ps_reg_01", DoubleType(), True),
    StructField("ps_reg_02", DoubleType(), True),
    StructField("ps_reg_03", DoubleType(), True),
    StructField("ps_car_01_cat", FloatType(), True),
    StructField("ps_car_02_cat", FloatType(), True),
    StructField("ps_car_03_cat", FloatType(), True),
    StructField("ps_car_04_cat", FloatType(), True),
    StructField("ps_car_05_cat", FloatType(), True),
    StructField("ps_car_06_cat", FloatType(), True),
    StructField("ps_car_07_cat", FloatType(), True),
    StructField("ps_car_08_cat", FloatType(), True),
    StructField("ps_car_09_cat", FloatType(), True),
    StructField("ps_car_10_cat", FloatType(), True),
    StructField("ps_car_11_cat", FloatType(), True),
    StructField("ps_car_12", FloatType(), True),
    StructField("ps_car_13", FloatType(), True),
    StructField("ps_car_14", FloatType(), True),
    StructField("ps_car_15", FloatType(), True),
    StructField("ps_calc_01", FloatType(), True),
    StructField("ps_calc_02", FloatType(), True),
    StructField("ps_calc_03", FloatType(), True),
    StructField("ps_calc_04", FloatType(), True),
    StructField("ps_calc_05", FloatType(), True),
    StructField("ps_calc_06", FloatType(), True),
    StructField("ps_calc_07", FloatType(), True),
    StructField("ps_calc_08", FloatType(), True),
    StructField("ps_calc_09", FloatType(), True),
    StructField("ps_calc_10", FloatType(), True),
    StructField("ps_calc_11", FloatType(), True),
    StructField("ps_calc_12", FloatType(), True),
    StructField("ps_calc_13", FloatType(), True),
    StructField("ps_calc_14", FloatType(), True),
    StructField("ps_calc_15_bin", FloatType(), True),
    StructField("ps_calc_16_bin", FloatType(), True),
    StructField("ps_calc_17_bin", FloatType(), True),
    StructField("ps_calc_18_bin", FloatType(), True),
    StructField("ps_calc_19_bin", FloatType(), True),
    StructField("ps_calc_20_bin", FloatType(), True)
])

# file ingestion
seguro = spark.read.csv("/FileStore/tables/train.csv",header=True, schema=schema)

'''
MissingValueHandler Estimator to replace -1 with the median value.
Mean will introduce impure values.
'''
imput = Imputer(inputCols=seguro.columns
                ,outputCols=seguro.columns)
imput.setMissingValue(-1)
imput.setStrategy("median")

'''
StringIndexer to encode a category to a indices. On Spark 2.2 we have to process separately each field
for version after we could use a single method call for all the fields.

'''
ps_car_01_cat_indexer = StringIndexer()
ps_car_01_cat_indexer.setInputCol("ps_car_01_cat")
ps_car_01_cat_indexer.setOutputCol("ps_car_01_cat_index")
ps_car_01_cat_indexer.setHandleInvalid("skip")

ps_car_02_cat_indexer = StringIndexer()
ps_car_02_cat_indexer.setInputCol("ps_car_02_cat")
ps_car_02_cat_indexer.setOutputCol("ps_car_02_cat_index")
ps_car_02_cat_indexer.setHandleInvalid("skip")

ps_car_03_cat_indexer = StringIndexer()
ps_car_03_cat_indexer.setInputCol("ps_car_03_cat")
ps_car_03_cat_indexer.setOutputCol("ps_car_03_cat_index")
ps_car_03_cat_indexer.setHandleInvalid("skip")

ps_car_04_cat_indexer = StringIndexer()
ps_car_04_cat_indexer.setInputCol("ps_car_04_cat")
ps_car_04_cat_indexer.setOutputCol("ps_car_04_cat_index")
ps_car_04_cat_indexer.setHandleInvalid("skip")

ps_car_05_cat_indexer = StringIndexer()
ps_car_05_cat_indexer.setInputCol("ps_car_05_cat")
ps_car_05_cat_indexer.setOutputCol("ps_car_05_cat_index")
ps_car_05_cat_indexer.setHandleInvalid("skip")

ps_car_06_cat_indexer = StringIndexer()
ps_car_06_cat_indexer.setInputCol("ps_car_06_cat")
ps_car_06_cat_indexer.setOutputCol("ps_car_06_cat_index")
ps_car_06_cat_indexer.setHandleInvalid("skip")

ps_car_07_cat_indexer = StringIndexer()
ps_car_07_cat_indexer.setInputCol("ps_car_07_cat")
ps_car_07_cat_indexer.setOutputCol("ps_car_07_cat_index")
ps_car_07_cat_indexer.setHandleInvalid("skip")

ps_car_08_cat_indexer = StringIndexer()
ps_car_08_cat_indexer.setInputCol("ps_car_08_cat")
ps_car_08_cat_indexer.setOutputCol("ps_car_08_cat_index")
ps_car_08_cat_indexer.setHandleInvalid("skip")

ps_car_09_cat_indexer = StringIndexer()
ps_car_09_cat_indexer.setInputCol("ps_car_09_cat")
ps_car_09_cat_indexer.setOutputCol("ps_car_09_cat_index")
ps_car_09_cat_indexer.setHandleInvalid("skip")

ps_car_10_cat_indexer = StringIndexer()
ps_car_10_cat_indexer.setInputCol("ps_car_10_cat")
ps_car_10_cat_indexer.setOutputCol("ps_car_10_cat_index")
ps_car_10_cat_indexer.setHandleInvalid("skip")

ps_ind_02_cat_indexer = StringIndexer()
ps_ind_02_cat_indexer.setInputCol("ps_ind_02_cat")
ps_ind_02_cat_indexer.setOutputCol("ps_ind_02_cat_index")
ps_ind_02_cat_indexer.setHandleInvalid("skip")

ps_ind_04_cat_indexer = StringIndexer()
ps_ind_04_cat_indexer.setInputCol("ps_ind_04_cat")
ps_ind_04_cat_indexer.setOutputCol("ps_ind_04_cat_index")
ps_ind_04_cat_indexer.setHandleInvalid("skip")

ps_ind_05_cat_indexer = StringIndexer()
ps_ind_05_cat_indexer.setInputCol("ps_ind_05_cat")
ps_ind_05_cat_indexer.setOutputCol("ps_ind_05_cat_index")
ps_ind_05_cat_indexer.setHandleInvalid("skip")
'''
OneHotEncoder transformer to map  
indices to a  binary vectors, where each vector.
'''
ps_car_01_cat_encoder = OneHotEncoder()
ps_car_01_cat_encoder.setInputCol("ps_car_01_cat_index")
ps_car_01_cat_encoder.setOutputCol("ps_car_01_cat_feature")

ps_car_02_cat_encoder = OneHotEncoder()
ps_car_02_cat_encoder.setInputCol("ps_car_02_cat_index")
ps_car_02_cat_encoder.setOutputCol("ps_car_02_cat_feature")

ps_car_03_cat_encoder = OneHotEncoder()
ps_car_03_cat_encoder.setInputCol("ps_car_03_cat_index")
ps_car_03_cat_encoder.setOutputCol("ps_car_03_cat_feature")

ps_car_04_cat_encoder = OneHotEncoder()
ps_car_04_cat_encoder.setInputCol("ps_car_04_cat_index")
ps_car_04_cat_encoder.setOutputCol("ps_car_04_cat_feature")

ps_car_05_cat_encoder = OneHotEncoder()
ps_car_05_cat_encoder.setInputCol("ps_car_05_cat_index")
ps_car_05_cat_encoder.setOutputCol("ps_car_05_cat_feature")

ps_car_06_cat_encoder = OneHotEncoder()
ps_car_06_cat_encoder.setInputCol("ps_car_06_cat_index")
ps_car_06_cat_encoder.setOutputCol("ps_car_06_cat_feature")

ps_car_07_cat_encoder = OneHotEncoder()
ps_car_07_cat_encoder.setInputCol("ps_car_07_cat_index")
ps_car_07_cat_encoder.setOutputCol("ps_car_07_cat_feature")

ps_car_08_cat_encoder = OneHotEncoder()
ps_car_08_cat_encoder.setInputCol("ps_car_08_cat_index")
ps_car_08_cat_encoder.setOutputCol("ps_car_08_cat_feature")

ps_car_09_cat_encoder = OneHotEncoder()
ps_car_09_cat_encoder.setInputCol("ps_car_09_cat_index")
ps_car_09_cat_encoder.setOutputCol("ps_car_09_cat_feature")

ps_car_10_cat_encoder = OneHotEncoder()
ps_car_10_cat_encoder.setInputCol("ps_car_10_cat_index")
ps_car_10_cat_encoder.setOutputCol("ps_car_10_cat_feature")

ps_ind_02_cat_encoder = OneHotEncoder()
ps_ind_02_cat_encoder.setInputCol("ps_ind_02_cat_index")
ps_ind_02_cat_encoder.setOutputCol("ps_ind_02_cat_feature")

ps_ind_04_cat_encoder = OneHotEncoder()
ps_ind_04_cat_encoder.setInputCol("ps_ind_04_cat_index")
ps_ind_04_cat_encoder.setOutputCol("ps_ind_04_cat_feature")

ps_ind_05_cat_encoder = OneHotEncoder()
ps_ind_05_cat_encoder.setInputCol("ps_ind_05_cat_index")
ps_ind_05_cat_encoder.setOutputCol("ps_ind_05_cat_feature")

'''
QuantileDiscretizer Estimator to discretize the continuous values into 10 bins
'''
ps_reg_01discretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_reg_01", outputCol="ps_reg_01_disc")
ps_reg_02discretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_reg_02", outputCol="ps_reg_02_disc")
ps_reg_03discretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_reg_03", outputCol="ps_reg_03_disc")
ps_car_11discretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_car_11_cat", outputCol="ps_car_11_disc")
ps_car_12discretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_car_12", outputCol="ps_car_12_disc") 
ps_car_13discretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_car_13", outputCol="ps_car_13_disc") 
ps_car_14discretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_car_14", outputCol="ps_car_14_disc") 
ps_car_15discretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_car_15", outputCol="ps_car_15_disc") 
ps_calc_01_catdiscretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_calc_01", outputCol="ps_calc_01_disc")  
ps_calc_02_catdiscretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_calc_02", outputCol="ps_calc_02_disc")  
ps_calc_03_catdiscretizer = QuantileDiscretizer(numBuckets=10, inputCol="ps_calc_03", outputCol="ps_calc_03_disc")


'''
OneHotEncoder to  one-hot encode vectors from these binned discretized values.
'''
ps_calc_01encoder = OneHotEncoder()
ps_calc_01encoder.setInputCol("ps_calc_01_disc")
ps_calc_01encoder.setOutputCol("ps_calc_01_disc_vec")

ps_calc_02encoder = OneHotEncoder()
ps_calc_02encoder.setInputCol("ps_calc_02_disc")
ps_calc_02encoder.setOutputCol("ps_calc_02_disc_vec")

ps_calc_03encoder = OneHotEncoder()
ps_calc_03encoder.setInputCol("ps_calc_03_disc")
ps_calc_03encoder.setOutputCol("ps_calc_03_disc_vec")
                                
ps_reg_01encoder = OneHotEncoder()
ps_reg_01encoder.setInputCol("ps_reg_01_disc")
ps_reg_01encoder.setOutputCol("ps_reg_01_disc_vec")

ps_reg_02encoder = OneHotEncoder()
ps_reg_02encoder.setInputCol("ps_reg_02_disc")
ps_reg_02encoder.setOutputCol("ps_reg_02_disc_vec")

ps_reg_03encoder = OneHotEncoder()
ps_reg_03encoder.setInputCol("ps_reg_03_disc")
ps_reg_03encoder.setOutputCol("ps_reg_03_disc_vec")   

ps_car_11encoder = OneHotEncoder()
ps_car_11encoder.setInputCol("ps_car_11_disc")
ps_car_11encoder.setOutputCol("ps_car_11_disc_vec")

ps_car_12encoder = OneHotEncoder()
ps_car_12encoder.setInputCol("ps_car_12_disc")
ps_car_12encoder.setOutputCol("ps_car_12_disc_vec")

ps_car_13encoder = OneHotEncoder()
ps_car_13encoder.setInputCol("ps_car_13_disc")
ps_car_13encoder.setOutputCol("ps_car_13_disc_vec")

ps_car_14encoder = OneHotEncoder()
ps_car_14encoder.setInputCol("ps_car_14_disc")
ps_car_14encoder.setOutputCol("ps_car_14_disc_vec")

ps_car_15encoder = OneHotEncoder()
ps_car_15encoder.setInputCol("ps_car_15_disc")
ps_car_15encoder.setOutputCol("ps_car_15_disc_vec")


'''
scaling and centere all the values with the standard scaler.
'''
scaler = StandardScaler()
scaler.setInputCol("features") 
scaler.setOutputCol("scaledFeatures")
scaler.setWithStd(True)
scaler.setWithMean(True)


pca = PCA() 
pca.setInputCol("scaledFeatures")
pca.setOutputCol("featurespca")
pca.setK(30)# similar result with 30 that with all, 25 a little less

pstages = [
 imput
,ps_car_01_cat_indexer
,ps_car_02_cat_indexer
,ps_car_03_cat_indexer
,ps_car_04_cat_indexer
,ps_car_05_cat_indexer
,ps_car_06_cat_indexer
,ps_car_07_cat_indexer
,ps_car_08_cat_indexer
,ps_car_09_cat_indexer
,ps_car_10_cat_indexer 
,ps_ind_02_cat_indexer
,ps_ind_04_cat_indexer
,ps_ind_05_cat_indexer  
,ps_car_01_cat_encoder
,ps_car_02_cat_encoder
,ps_car_03_cat_encoder
,ps_car_04_cat_encoder
,ps_car_05_cat_encoder
,ps_car_06_cat_encoder
,ps_car_07_cat_encoder
,ps_car_08_cat_encoder
,ps_car_09_cat_encoder
,ps_car_10_cat_encoder 
,ps_ind_02_cat_encoder
,ps_ind_04_cat_encoder
,ps_ind_05_cat_encoder   
,ps_reg_01discretizer
,ps_reg_02discretizer
,ps_reg_03discretizer
,ps_car_11discretizer 
,ps_car_12discretizer 
,ps_car_13discretizer 
,ps_car_14discretizer 
,ps_car_15discretizer 
,ps_calc_01_catdiscretizer
,ps_calc_02_catdiscretizer
,ps_calc_03_catdiscretizer
,ps_calc_01encoder
,ps_calc_02encoder
,ps_calc_03encoder
,ps_reg_01encoder
,ps_reg_02encoder
,ps_reg_03encoder
,ps_car_11encoder
,ps_car_12encoder
,ps_car_13encoder
,ps_car_14encoder
,ps_car_15encoder
,assembler
,scaler
,pca]

trainingpipeline = Pipeline()

#stratification test ratio is 1: 0.0364 and 0: 0.964
stratified_data = seguro.sampleBy('target', fractions={0: 0.0364, 1: 1.0}).cache()

[train,test] = stratified_data.randomSplit([0.67, 0.33])

lr = LogisticRegression()
lr.setMaxIter(30)
lr.setFeaturesCol("featurespca")
lr.setLabelCol("target")
lr.setFamily("binomial")

trainingpipeline.setStages(pstages + [lr])

linear_model = trainingpipeline.fit(train)


rf = RandomForestClassifier(numTrees=30)
rf.setFeaturesCol("featurespca")
rf.setLabelCol("target")

trainingpipeline.setStages(pstages + [rf])

rf_model = trainingpipeline.fit(train)

gbt = GBTClassifier(maxIter=30)
gbt.setFeaturesCol("featurespca")
gbt.setLabelCol("target")


trainingpipeline.setStages(pstages + [gbt])

gbt_model = trainingpipeline.fit(train)

from pyspark.ml.classification import LinearSVC
svm = LinearSVC(maxIter=30)
svm.setFeaturesCol("featurespca")
svm.setLabelCol("target")

trainingpipeline.setStages(pstages + [svm])

svm_model = trainingpipeline.fit(train)

linear_testDataPredictions = linear_model.transform(test)
rf_testDataPredictions = rf_model.transform(test)
gbt_testDataPredictions = gbt_model.transform(test)

evaluator = MulticlassClassificationEvaluator()
'''
We utilize the MulticlassClassificationEvaluator for computing
the evaluation.
'''
evaluator = MulticlassClassificationEvaluator()
evaluator.setLabelCol("target")
evaluator.setPredictionCol("prediction")
evaluator.setMetricName("accuracy")

accuracy = evaluator.evaluate(linear_testDataPredictions)
print("Accuracy lr: {0}".format(accuracy))

accuracy = evaluator.evaluate(rf_testDataPredictions)
print("Accuracy rf: {0}".format(accuracy))

# note that in this version of Spark, Gradient Boosted Decision Trees don't support outputing raw predictions, only thresholded ones
accuracy = evaluator.evaluate(gbt_testDataPredictions)
print("Accuracy gbt: {0}".format(accuracy))

accuracy = evaluator.evaluate(svm_testDataPredictions)
print("Accuracy svm: {0}".format(accuracy))

evaluator = MulticlassClassificationEvaluator()
evaluator.setLabelCol("target")
evaluator.setPredictionCol("prediction")
evaluator.setMetricName("f1")

f1 = evaluator.evaluate(linear_testDataPredictions)
print("f1 lr: {0}".format(f1))

f1 = evaluator.evaluate(rf_testDataPredictions)
print("f1 rf: {0}".format(f1))

# note that in this version of Spark, Gradient Boosted Decision Trees don't support outputing raw predictions, only thresholded ones
f1 = evaluator.evaluate(gbt_testDataPredictions)
print("f1 gbt: {0}".format(f1))

f1 = evaluator.evaluate(svm_testDataPredictions)
print("f1 svm: {0}".format(f1))

evaluator = MulticlassClassificationEvaluator()
evaluator.setLabelCol("target")
evaluator.setPredictionCol("prediction")
evaluator.setMetricName("weightedPrecision")

weightedPrecision = evaluator.evaluate(linear_testDataPredictions)
print("weightedPrecision lr: {0}".format(weightedPrecision))

weightedPrecision = evaluator.evaluate(rf_testDataPredictions)
print("weightedPrecision rf: {0}".format(weightedPrecision))

# note that in this version of Spark, Gradient Boosted Decision Trees don't support outputing raw predictions, only thresholded ones
weightedPrecision = evaluator.evaluate(gbt_testDataPredictions)
print("weightedPrecision gbt: {0}".format(weightedPrecision))

weightedPrecision = evaluator.evaluate(svm_testDataPredictions)
print("weightedPrecision svm: {0}".format(weightedPrecision))

evaluator = MulticlassClassificationEvaluator()
evaluator.setLabelCol("target")
evaluator.setPredictionCol("prediction")
evaluator.setMetricName("weightedRecall")

weightedRecall = evaluator.evaluate(linear_testDataPredictions)
print("weightedRecall lr: {0}".format(weightedRecall))

weightedRecall = evaluator.evaluate(rf_testDataPredictions)
print("weightedRecall rf: {0}".format(weightedRecall))

# note that in this version of Spark, Gradient Boosted Decision Trees don't support outputing raw predictions, only thresholded ones
weightedRecall = evaluator.evaluate(gbt_testDataPredictions)
print("weightedRecall gbt: {0}".format(weightedRecall))

# note that in this version of Spark, Gradient Boosted Decision Trees don't support outputing raw predictions, only thresholded ones
weightedRecall = evaluator.evaluate(svm_testDataPredictions)
print("weightedRecall svm: {0}".format(weightedRecall))

'''
We utilize the BinaryClassificationEvaluator for computing
the evaluation. This evaluator by default computes the
'areaUnderROC'. One can also choose to compute
the 'areaUnderPR' curve by setting the metric using the
setMetricName method
'''
evaluator = BinaryClassificationEvaluator()
evaluator.setLabelCol("target")
evaluator.setRawPredictionCol("rawPrediction")
evaluator.setMetricName("areaUnderROC")

lr_roc_test = evaluator.evaluate(linear_testDataPredictions)
print("Test Linear ROC: {0:.5f}".format(lr_roc_test))

rf_roc_test = evaluator.evaluate(rf_testDataPredictions)
print("Test Random Forest ROC: {0:.5f}".format(rf_roc_test))

gbt_roc_test = evaluator.evaluate(gbt_testDataPredictions)
print("Test GBT ROC: {0:.5f}".format(gbt_roc_test))

svm_roc_test = evaluator.evaluate(svm_testDataPredictions)
print("Test SVM ROC: {0:.5f}".format(svm_roc_test))

predresscore = gbt_testDataPredictions.withColumn("score",scorepickfuncudf(gbt_testDataPredictions["probability"]))

'''
We compute the false positive rate and true positive rate at various thresholds
of the probability score and use that to recompute the auc and finally to 
plot the ROC curve.
'''
false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probs)
roc_auc = auc(false_positive_rate, true_positive_rate)

fig, ax = plt.subplots()
plt.title('GBT Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

display(fig)

#TODO do it for the best algorithm

paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

cross_val = CrossValidator(estimator=trainingpipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=2)  

model = cross_val.fit(train)
    
results = model.transform(test) 

# Compute area under ROC score
predictionLabels = results.select("prediction", "label")
metrics = BinaryClassificationMetrics(predictionLabels.rdd)
metrics.areaUnderROC    


paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()
    
evaluator = BinaryClassificationEvaluator()
evaluator.setLabelCol("target")
evaluator.setRawPredictionCol("rawPrediction")
evaluator.setMetricName("areaUnderROC")   

pipeline = trainingpipeline.setStages(pstages + [lr])    
    
cross_val = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator= evaluator,
                          numFolds=5)  

cv_model = cross_val.fit(train) 

bestModel = cv_model.bestModel

bestModel.save("lrSeguro20180330")

#bestModel = LogisticRegression.load("lrSeguro20180330")

best_testDataPredictions = bestModel.transform(test)

evaluator = BinaryClassificationEvaluator()
evaluator.setLabelCol("target")
evaluator.setRawPredictionCol("rawPrediction")
evaluator.setMetricName("areaUnderROC")

lr_roc_test = evaluator.evaluate(best_testDataPredictions)
print("Test Linear ROC: {0:.5f}".format(lr_roc_test))

evaluator = MulticlassClassificationEvaluator()
evaluator.setLabelCol("target")
evaluator.setPredictionCol("prediction")
evaluator.setMetricName("weightedRecall")

weightedRecall = evaluator.evaluate(linear_testDataPredictions)
print("weightedRecall lr: {0}".format(weightedRecall))

weightedRecall = evaluator.evaluate(linear_testDataPredictions)
print("weightedRecall lr: {0}".format(weightedPrecision))

evaluator.setMetricName("weightedPrecision")

weightedRecall = evaluator.evaluate(linear_testDataPredictions)
print("weightedRecall lr: {0}".format(f1))

evaluator.setMetricName("f1")

accuracy = evaluator.evaluate(best_testDataPredictions)
print("Accuracy lr: {0}".format(accuracy))
