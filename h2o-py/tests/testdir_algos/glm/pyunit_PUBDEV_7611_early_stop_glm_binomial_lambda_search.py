from __future__ import division
from __future__ import print_function
from past.utils import old_div
import sys
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

# In this test, we check and make sure that GAM can do early stopping
def test_glm_earlyStop():
    early_stop_metrics = ["logloss", "deviance", "r2"]
    max_stopping_rounds = 3            # maximum stopping rounds allowed to be used for early stopping metric
    max_tolerance = 0.01                # maximum tolerance to be used for early stopping metric
    
    print("Preparing dataset....")
    h2o_data = h2o.import_file(pyunit_utils.locate("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    h2o_data["C3"] = h2o_data["C3"].asfactor()
    h2o_data["C4"] = h2o_data["C4"].asfactor()
    h2o_data["C5"] = h2o_data["C5"].asfactor()
    h2o_data["C11"] = h2o_data["C11"].asfactor()
    splits <- h2o.splitFrame(data =  h2o_data, ratios = .8, seed = 1234)
    train <- splits[[1]]
    valid <- splits[[2]]
    
    print("Start earlying stopping test")
    h2o_model_no_early_stop = glm(family="multinomial")
    h2o_model_no_early_stop.train(x=[0,1,2,3,4,5,6,7,8,9], y="C11", training_frame=h2o_data) # model with no early stop
    for ind in range(len(early_stop_metrics)):
        h2o_model = glm(family="multinomial", stopping_rounds=max_stopping_rounds, score_each_iteration=True,
                        stopping_metric=early_stop_metrics[ind], stopping_tolerance=max_tolerance)
        h2o_model.train(x=[0,1,2,3,4,5,6,7,8,9], y="C11", training_frame=h2o_data)
        h2o_model2 = glm(family="multinomial", stopping_rounds=max_stopping_rounds, score_iteration_interval=1,
                        stopping_metric=early_stop_metrics[ind], stopping_tolerance=max_tolerance)
        h2o_model2.train(x=[0,1,2,3,4,5,6,7,8,9], y="C11", training_frame=h2o_data)
        assert h2o_model.coeff()==h2o_model2.coeff()
        print("Done")
        
        

if __name__ == "__main__":
    pyunit_utils.standalone_test(test_glm_earlyStop)
else:
    test_glm_earlyStop()
