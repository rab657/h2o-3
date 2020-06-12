from __future__ import division
from __future__ import print_function
from past.utils import old_div
import sys
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator

# In this test, we check and make sure that GAM can do early stopping
def test_gam_gamColumns():
    early_stop_metrics = ["logloss", "deviance", "RMSE"]
    max_stopping_rounds = 5            # maximum stopping rounds allowed to be used for early stopping metric
    max_tolerance = 0.01                # maximum tolerance to be used for early stopping metric
    
    print("Preparing dataset....")
    h2o_data = h2o.import_file(pyunit_utils.locate("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    myX = ["C1", "C2"]
    myY = "C11"
    h2o_data["C11"] = h2o_data["C11"].asfactor()
    print("Start earlying stopping test")
    for ind in range(len(early_stop_metrics)):
        h2o_model = H2OGeneralizedAdditiveEstimator(family="multinomial", gam_columns=["C6", "C7", "C8"],
                                                    keep_gam_cols=True, scale = [1,1,1], num_knots=[5,5,5], )
        h2o_model.train(x=myX, y=myY, training_frame=h2o_data)
        
    gamFrame = h2o.get_frame(h2o_model._model_json["output"]["gam_transformed_center_key"])
    gamFrame = gamFrame.drop("C1").drop("C2").drop("C11")
    gamFrameAns = h2o.import_file(pyunit_utils.locate("smalldata/gam_test/multinomial_10_classes_10_cols_10000_Rows_train_C6Gam_center.csv"))
    gamFrameAns = gamFrameAns.cbind (h2o.import_file(pyunit_utils.locate("smalldata/gam_test/multinomial_10_classes_10_cols_10000_Rows_train_C7Gam_center.csv")))
    gamFrameAns = gamFrameAns.cbind(h2o.import_file(pyunit_utils.locate("smalldata/gam_test/multinomial_10_classes_10_cols_10000_Rows_train_C8Gam_center.csv")))
    pyunit_utils.compare_frames_local(gamFrameAns, gamFrame)
    print("gam gamcolumn test completed successfully")

if __name__ == "__main__":
    pyunit_utils.standalone_test(test_gam_gamColumns)
else:
    test_gam_gamColumns()
