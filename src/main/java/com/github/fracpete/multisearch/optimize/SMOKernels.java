/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * J48ConfidenceFactor.java
 * Copyright (C) 2016 University of Waikato, Hamilton, NZ
 */

package com.github.fracpete.multisearch.optimize;

import com.github.fracpete.multisearch.ExampleHelper;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.MultiSearch;
import weka.classifiers.meta.multisearch.DefaultEvaluationMetrics;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.ListParameter;
import weka.core.setupgenerator.MathParameter;
import weka.core.setupgenerator.ParameterGroup;

/**
 * Optimizes the RBFKernel and PolyKernel for SMO in separate search-spaces.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class SMOKernels {

  /**
   * The first parameter must be dataset,
   * the (optional) second the class index (1-based, 'first' and 'last'
   * also supported).
   *
   * @param args	the commandline options
   * @throws Exception	if optimization fails for some reason
   */
  public static void main(String[] args) throws Exception {
    if (args.length == 0) {
      System.err.println("\nUsage: SMOKernels <dataset> [classindex]\n");
      System.exit(1);
    }

    // load data
    Instances data = ExampleHelper.loadData(args[0], (args.length > 1) ? args[1] : null);

    // configure classifier we want to optimize
    SMO smo = new SMO();

    // configure multisearch
    // 1. RBFKernel
    ListParameter listRBF = new ListParameter();
    listRBF.setProperty("kernel");
    listRBF.setList(RBFKernel.class.getName());
    MathParameter gamma = new MathParameter();
    gamma.setProperty("kernel.gamma");
    gamma.setBase(10);
    gamma.setMin(-4);
    gamma.setMax(1);
    gamma.setStep(1);
    gamma.setExpression("pow(BASE,I)");
    ParameterGroup groupRBF = new ParameterGroup();
    groupRBF.setParameters(new AbstractParameter[]{
      listRBF,
      gamma
    });
    // 2. PolyKernel
    ListParameter listPoly = new ListParameter();
    listPoly.setProperty("kernel");
    listPoly.setList(PolyKernel.class.getName());
    MathParameter exp = new MathParameter();
    exp.setProperty("kernel.exponent");
    exp.setBase(10);
    exp.setMin(1);
    exp.setMax(5);
    exp.setStep(1);
    exp.setExpression("I");
    ParameterGroup groupPoly = new ParameterGroup();
    groupPoly.setParameters(new AbstractParameter[]{
      listPoly,
      exp
    });
    // assemble everything
    MultiSearch multi = new MultiSearch();
    multi.setClassifier(smo);
    multi.setSearchParameters(new AbstractParameter[]{
      groupRBF,
      groupPoly
    });
    SelectedTag tag = new SelectedTag(
      DefaultEvaluationMetrics.EVALUATION_ACC,
      new DefaultEvaluationMetrics().getTags());
    multi.setEvaluation(tag);

    // output configuration
    System.out.println("\nMultiSearch commandline:\n" + Utils.toCommandLine(multi));

    // optimize
    System.out.println("\nOptimizing...\n");
    multi.buildClassifier(data);
    System.out.println("Best setup:\n" + Utils.toCommandLine(multi.getBestClassifier()));
    System.out.println("Best parameters: " + multi.getGenerator().evaluate(multi.getBestValues()));
  }
}
