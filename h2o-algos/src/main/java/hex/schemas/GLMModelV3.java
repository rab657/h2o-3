package hex.schemas;

import hex.ModelCategory;
import hex.ScoreKeeper;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMOutput;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import water.MemoryManager;
import water.api.API;
import water.api.schemas3.ModelOutputSchemaV3;
import water.api.schemas3.ModelSchemaV3;
import water.api.schemas3.TwoDimTableV3;
import water.util.ArrayUtils;
import water.util.PrettyPrint;
import water.util.TwoDimTable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
//import water.util.DocGen.HTML;

public class GLMModelV3 extends ModelSchemaV3<GLMModel, GLMModelV3, GLMModel.GLMParameters, GLMV3.GLMParametersV3, GLMOutput, GLMModelV3.GLMModelOutputV3> {

  public static final class GLMModelOutputV3 extends ModelOutputSchemaV3<GLMOutput, GLMModelOutputV3> {

    @API(help="Table of Coefficients")
    TwoDimTableV3 coefficients_table;

    @API(help="Table of Random Coefficients for HGLM")
    TwoDimTableV3 random_coefficients_table;

    @API(help="Table of Scoring History for Early Stop")
    TwoDimTable scoring_history_early_stop;

    @API(help="Table of Coefficients with coefficients denoted with class names for GLM multinonimals only.")
    TwoDimTableV3 coefficients_table_multinomials_with_class_names;  // same as coefficients_table but with real class names.

    @API(help="Standardized Coefficient Magnitudes")
    TwoDimTableV3 standardized_coefficient_magnitudes;

    @API(help="Lambda minimizing the objective value, only applicable with lambd search")
    double lambda_best;

    @API(help="Lambda best + 1 standard error. Only applicable with lambda search and cross-validation")
    double lambda_1se;

    @API(help = "Dispersion parameter, only applicable to Tweedie family (input/output) and fractional Binomial (output only)")
    double dispersion;

    private GLMModelOutputV3 fillMultinomial(GLMOutput impl) {
      if(impl.get_global_beta_multinomial() == null)
        return this; // no coefificients yet
      String [] names = impl.coefficientNames().clone();
      // put intercept as the first
      String [] ns = ArrayUtils.append(new String[]{"Intercept"},Arrays.copyOf(names,names.length-1));

      coefficients_table = new TwoDimTableV3();
      if (impl.nclasses() > 2) // only change coefficient names for multinomials
        coefficients_table_multinomials_with_class_names = new TwoDimTableV3();
      
        int n = impl.nclasses();
        String[] cols = new String[n*2];
        String[] cols2=null;
        if (n>2) {
          cols2 = new String[n*2];
          String[] classNames = impl._domains[impl.responseIdx()];
          for (int i = 0; i < n; ++i) {
            cols2[i] = "coefs_class_" + classNames[i];
            cols2[n + i] = "std_coefs_class_" + classNames[i];
          }
        }
        for (int i = 0; i < n; ++i) {
          cols[i] = "coefs_class_" +i;
          cols[n + i] = "std_coefs_class_" +i;
        }

        String [] colTypes = new String[cols.length];
        Arrays.fill(colTypes, "double");
        String [] colFormats = new String[cols.length];
        Arrays.fill(colFormats,"%5f");

        double [][] betaNorm = impl.getNormBetaMultinomial();
        if(betaNorm != null) {
          TwoDimTable tdt = new TwoDimTable("Coefficients", "glm multinomial coefficients", ns, cols, colTypes, colFormats, "names");
          for (int c = 0; c < n; ++c) {
            double[] beta = impl.get_global_beta_multinomial()[c];
            tdt.set(0, c, beta[beta.length - 1]);
            tdt.set(0, n + c, betaNorm[c][beta.length - 1]);
            for (int i = 0; i < beta.length - 1; ++i) {
              tdt.set(i + 1, c, beta[i]);
              tdt.set(i + 1, n + c, betaNorm[c][i]);
            }
          }
          coefficients_table.fillFromImpl(tdt);

          if (n>2) {  // restore column names from pythonized ones
            coefficients_table_multinomials_with_class_names.fillFromImpl(tdt);
            revertCoeffNames(cols2, n, coefficients_table_multinomials_with_class_names);
          }
          final double [] magnitudes = new double[betaNorm[0].length];
          for(int i = 0; i < betaNorm.length; ++i) {
            for (int j = 0; j < betaNorm[i].length; ++j) {
              double d = betaNorm[i][j];
              magnitudes[j] += d < 0 ? -d : d;
            }
          }
          Integer [] indices = new Integer[magnitudes.length-1];
          for(int i = 0; i < indices.length; ++i)
            indices[i] = i;
          Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
              if(magnitudes[o1] < magnitudes[o2]) return +1;
              if(magnitudes[o1] > magnitudes[o2]) return -1;
              return 0;
            }
          });
          int len = names.length-1;
          String [] names2 = new String[len]; // this one decides the length of standardized table length
          for(int i = 0; i < len; ++i)
            names2[i] = names[indices[i]];
          tdt = new TwoDimTable("Standardized Coefficient Magnitudes", "standardized coefficient magnitudes", names2, new String[]{"Coefficients", "Sign"}, new String[]{"double", "string"}, new String[]{"%5f", "%s"}, "names");
          for (int i = 0; i < magnitudes.length - 1; ++i) {
            tdt.set(i, 0, magnitudes[indices[i]]);
            tdt.set(i, 1, "POS");
          }
          standardized_coefficient_magnitudes = new TwoDimTableV3();
          standardized_coefficient_magnitudes.fillFromImpl(tdt);
        }
      scoring_history_early_stop = createScoringHistoryTable(impl, impl._validation_metrics!=null,
              impl._cross_validation_metrics != null, impl.getModelCategory());
      return this;
    }

    public void revertCoeffNames(String[] colNames, int nclass, TwoDimTableV3 coeffs_table) {
      String newName = coeffs_table.name+" with class names";
      coeffs_table.name = newName;
      boolean bothCoeffStd = colNames.length==(2*nclass);
      for (int tableIndex = 1; tableIndex <= nclass;  tableIndex++) {
        coeffs_table.columns[tableIndex].name = new String(colNames[tableIndex-1]);
        if (bothCoeffStd)
          coeffs_table.columns[tableIndex+nclass].name = new String(colNames[tableIndex-1+nclass]);
      }
    }
    
    public TwoDimTable buildRandomCoefficients2DTable(double[] ubeta, String[] randomColNames) {
      String [] colTypes = new String[]{"double"};
      String [] colFormats = new String[]{"%5f"};
      String [] colnames = new String[]{"Random Coefficients"};
      TwoDimTable tdt = new TwoDimTable("HGLM Random Coefficients",
              "HGLM random coefficients", randomColNames, colnames, colTypes, colFormats,
              "names");
      // fill in coefficients
      for (int i = 0; i < ubeta.length; ++i) {
        tdt.set(i, 0, ubeta[i]);
      }
      return tdt;
    }

    @Override
    public GLMModelOutputV3 fillFromImpl(GLMModel.GLMOutput impl) {
      super.fillFromImpl(impl);
      lambda_1se = impl.lambda_1se();
      lambda_best = impl.lambda_best();
      dispersion = impl.dispersion();
      
      if(impl._multinomial || impl._ordinal)
        return fillMultinomial(impl);
      String [] names = impl.coefficientNames().clone();
      // put intercept as the first
      String [] ns = ArrayUtils.append(new String[]{"Intercept"},Arrays.copyOf(names,names.length-1));
      coefficients_table = new TwoDimTableV3();
      if ((impl.ubeta() != null) && (impl.randomcoefficientNames()!= null)) {
        random_coefficients_table = new TwoDimTableV3();
        random_coefficients_table.fillFromImpl(buildRandomCoefficients2DTable(impl.ubeta(), impl.randomcoefficientNames()));
      }
      final double [] magnitudes;
      double [] beta = impl.beta();
      if(beta == null) beta = MemoryManager.malloc8d(names.length);
      String [] colTypes = new String[]{"double"};
      String [] colFormats = new String[]{"%5f"};
      String [] colnames = new String[]{"Coefficients"};

      if(impl.hasPValues()){
        colTypes = new String[]{"double","double","double","double"};
        colFormats = new String[]{"%5f","%5f","%5f","%5f"};
        colnames = new String[]{"Coefficients","Std. Error","z value","p value"};
      }
      int stdOff = colnames.length;
      colTypes = ArrayUtils.append(colTypes,"double");
      colFormats = ArrayUtils.append(colFormats,"%5f");
      colnames = ArrayUtils.append(colnames,"Standardized Coefficients");
      TwoDimTable tdt = new TwoDimTable("Coefficients","glm coefficients", ns, colnames, colTypes, colFormats, "names");
      tdt.set(0, 0, beta[beta.length - 1]);
      for (int i = 0; i < beta.length - 1; ++i) {
        tdt.set(i + 1, 0, beta[i]);
      }
      double[] norm_beta = null;
      if(impl.beta() != null) {
        norm_beta = impl.getNormBeta();
        tdt.set(0, stdOff, norm_beta[norm_beta.length - 1]);
        for (int i = 0; i < norm_beta.length - 1; ++i)
          tdt.set(i + 1, stdOff, norm_beta[i]);
      }
      if(impl.hasPValues()) { // fill in p values
        double [] stdErr = impl.stdErr();
        double [] zVals = impl.zValues();
        double [] pVals = impl.pValues();
        tdt.set(0, 1, stdErr[stdErr.length - 1]);
        tdt.set(0, 2, zVals[zVals.length - 1]);
        tdt.set(0, 3, pVals[pVals.length - 1]);
        for(int i = 0; i < stdErr.length - 1; ++i) {
          tdt.set(i + 1, 1, stdErr[i]);
          tdt.set(i + 1, 2, zVals[i]);
          tdt.set(i + 1, 3, pVals[i]);
        }
      }
      coefficients_table.fillFromImpl(tdt);
      if(impl.beta() != null) {
        magnitudes = norm_beta.clone();
        for (int i = 0; i < magnitudes.length; ++i)
          if (magnitudes[i] < 0) magnitudes[i] *= -1;
        Integer[] indices = new Integer[magnitudes.length - 1];
        for (int i = 0; i < indices.length; ++i)
          indices[i] = i;
        Arrays.sort(indices, new Comparator<Integer>() {
          @Override
          public int compare(Integer o1, Integer o2) {
            if (magnitudes[o1] < magnitudes[o2]) return +1;
            if (magnitudes[o1] > magnitudes[o2]) return -1;
            return 0;
          }
        });
        int len = names.length-1;
        String[] names2 = new String[len];
        for (int i = 0; i < len; ++i)
          names2[i] = names[indices[i]];
        tdt = new TwoDimTable("Standardized Coefficient Magnitudes", "standardized coefficient magnitudes", names2, new String[]{"Coefficients", "Sign"}, new String[]{"double", "string"}, new String[]{"%5f", "%s"}, "names");
        for (int i = 0; i < beta.length - 1; ++i) {
          tdt.set(i, 0, magnitudes[indices[i]]);
          tdt.set(i, 1, beta[indices[i]] < 0 ? "NEG" : "POS");
        }
        standardized_coefficient_magnitudes = new TwoDimTableV3();
        standardized_coefficient_magnitudes.fillFromImpl(tdt);
      }
      if (impl._scored_train.size() > 0)
        scoring_history_early_stop = createScoringHistoryTable(impl, impl._validation_metrics!=null, 
                impl._cross_validation_metrics != null, impl.getModelCategory());
      return this;
    }
  } // GLMModelOutputV2

// create scoring history table using scoreKeeper arrays scored_train, scored_xval, scored_valid
  public static TwoDimTable createScoringHistoryTable(GLMOutput glmOutput, boolean hasValidation, 
                                                      boolean hasCrossValidation, ModelCategory modelCategory) {
    boolean isClassifier = (modelCategory == ModelCategory.Binomial || modelCategory == ModelCategory.Multinomial
            || modelCategory == ModelCategory.Ordinal);
    List<String> colHeaders = new ArrayList<>();
    List<String> colTypes = new ArrayList<>();
    List<String> colFormat = new ArrayList<>();
    colHeaders.add("Timestamp"); colTypes.add("string"); colFormat.add("%s");
    colHeaders.add("Duration"); colTypes.add("string"); colFormat.add("%s");
    colHeaders.add("Evaluation_Iterations"); colTypes.add("int"); colFormat.add("%d");
    colHeaders.add("Training RMSE"); colTypes.add("double"); colFormat.add("%.5f");
    setColHeader(colHeaders, colTypes, colFormat, modelCategory, isClassifier, "Training");
    if (hasValidation)
      setColHeader(colHeaders, colTypes, colFormat, modelCategory, isClassifier, "Validation");
    if (hasCrossValidation)
      setColHeader(colHeaders, colTypes, colFormat, modelCategory, isClassifier, "Cross-Validation");
 
    final int rows = glmOutput._scored_train == null ? 0 : glmOutput._scored_train.size();
    String[] s = new String[0];
    TwoDimTable table = new TwoDimTable(
            "Scoring History", null,
            new String[rows],
            colHeaders.toArray(s),
            colTypes.toArray(s),
            colFormat.toArray(s),
            "");
    int row = 0;
    if (null == glmOutput._scored_train)
      return table;
    int iteration = 0;
    for (ScoreKeeper si : glmOutput._scored_train) { // fill out the table
      int col = 0;
      assert (row < table.getRowDim());
      assert (col < table.getColDim());
      DateTimeFormatter fmt = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss");
      long training_time_ms = glmOutput._training_time_ms.get(row);
      table.set(row, col++, fmt.print(training_time_ms));
      table.set(row, col++, PrettyPrint.msecs(training_time_ms-glmOutput._start_time, true));
      table.set(row, col++, iteration);
      col = setTableEntry(table, si, row, col, modelCategory, isClassifier); // entry for training dataset
      if (hasValidation && (glmOutput._scored_valid.size() > 0))
        col = setTableEntry(table, glmOutput._scored_valid.get(row), row, col, modelCategory, isClassifier);
      if (hasCrossValidation  && (glmOutput._scored_xval.size() > 0))
        col = setTableEntry(table, glmOutput._scored_xval.get(row), row, col, modelCategory, isClassifier);
      row++;
    }
    return table;
  }

  public static void setColHeader(List<String> colHeaders, List<String> colTypes, List<String> colFormat, 
                                  ModelCategory modelCategory, boolean isClassifier, String metricType) {
    colHeaders.add(metricType+"_RMSE"); colTypes.add("double"); colFormat.add("%.5f");
    if (modelCategory == ModelCategory.Regression) {
      colHeaders.add(metricType+" Deviance"); colTypes.add("double"); colFormat.add("%.5f");
      colHeaders.add(metricType+" MAE"); colTypes.add("double"); colFormat.add("%.5f");
      colHeaders.add(metricType+" r2"); colTypes.add("double"); colFormat.add("%.5f");
    }
    if (isClassifier) {
      colHeaders.add(metricType+" LogLoss"); colTypes.add("double"); colFormat.add("%.5f");
      colHeaders.add(metricType+" r2"); colTypes.add("double"); colFormat.add("%.5f");
    }
    if (modelCategory == ModelCategory.Binomial) {
      colHeaders.add(metricType+" AUC"); colTypes.add("double"); colFormat.add("%.5f");
      colHeaders.add(metricType+" pr_auc"); colTypes.add("double"); colFormat.add("%.5f");
      colHeaders.add(metricType+" Lift"); colTypes.add("double"); colFormat.add("%.5f");
    }
    if (isClassifier) {
      colHeaders.add(metricType+" Classification Error"); colTypes.add("double"); colFormat.add("%.5f");
    }
  }
  
  public static int setTableEntry(TwoDimTable table, ScoreKeeper scored_metric, int row, int col,
                                  ModelCategory modelCategory, boolean isClassifier) {
    table.set(row, col++, scored_metric != null ? scored_metric._rmse : Double.NaN);
    if (modelCategory == ModelCategory.Regression) {
      table.set(row, col++, scored_metric != null ? scored_metric._mean_residual_deviance : Double.NaN);
      table.set(row, col++, scored_metric != null ? scored_metric._mae : Double.NaN);
      table.set(row, col++, scored_metric != null ? scored_metric._r2 : Double.NaN);
    }
    if (isClassifier) {
      table.set(row, col++, scored_metric != null ? scored_metric._logloss : Double.NaN);
      table.set(row, col++, scored_metric != null ? scored_metric._r2 : Double.NaN);
    }
    if (modelCategory == ModelCategory.Binomial) {
      table.set(row, col++, scored_metric != null ? scored_metric._AUC : Double.NaN);
      table.set(row, col++, scored_metric != null ? scored_metric._pr_auc : Double.NaN);
      table.set(row, col++, scored_metric != null ? scored_metric._lift : Double.NaN);
    }
    if (isClassifier) {
      table.set(row, col, scored_metric != null ? scored_metric._classError : Double.NaN);
    }

    return col;
  }

  public GLMV3.GLMParametersV3 createParametersSchema() { return new GLMV3.GLMParametersV3(); }
  public GLMModelOutputV3 createOutputSchema() { return new GLMModelOutputV3(); }

  @Override public GLMModel createImpl() {
    GLMModel.GLMParameters parms = parameters.createImpl();
    return new GLMModel( model_id.key(), parms, null, new double[]{0.0}, 0.0, 0.0, 0);
  }
}
