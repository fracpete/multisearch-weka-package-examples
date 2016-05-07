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
import weka.classifiers.meta.MultiSearch;
import weka.classifiers.meta.multisearch.DefaultEvaluationMetrics;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.MathParameter;

/**
 * Optimizes the confidence factor of the J48 classifier.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class J48ConfidenceFactor {

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
      System.err.println("\nUsage: J48ConfidenceFactor <dataset> [classindex]\n");
      System.exit(1);
    }

    // load data
    Instances data = ExampleHelper.loadData(args[0], (args.length > 1) ? args[1] : null);

    // configure classifier we want to optimize
    J48 j48 = new J48();

    // configure multisearch
    MathParameter conf = new MathParameter();
    conf.setProperty("confidenceFactor");
    conf.setBase(10);
    conf.setMin(0.05);
    conf.setMax(0.75);
    conf.setStep(0.05);
    conf.setExpression("I");
    MultiSearch multi = new MultiSearch();
    multi.setClassifier(j48);
    multi.setSearchParameters(new AbstractParameter[]{
      conf
    });
    SelectedTag tag = new SelectedTag(
      DefaultEvaluationMetrics.EVALUATION_AUC,
      new DefaultEvaluationMetrics().getTags());
    multi.setEvaluation(tag);

    // output configuration
    System.out.println("\nMultiSearch commandline:\n" + Utils.toCommandLine(multi));

    // optimize
    System.out.println("\nOptimizing...\n");
    multi.buildClassifier(data);
    System.out.println("Best setup:\n" + Utils.toCommandLine(multi.getBestClassifier()));
    System.out.println("Best parameter: " + multi.getGenerator().evaluate(multi.getBestValues()));
  }
}
