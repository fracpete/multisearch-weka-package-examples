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
 * PLSFilterAndLinearRegression.java
 * Copyright (C) 2015 University of Waikato, Hamilton, NZ
 */

package com.github.fracpete.multisearch.optimize;

import com.github.fracpete.multisearch.ExampleHelper;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.MultiSearch;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.ListParameter;
import weka.core.setupgenerator.MathParameter;
import weka.filters.supervised.attribute.PLSFilter;

/**
 * Optimizes the number of components of PLSFilter and the ridge
 * parameter of LinearRegression.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class PLSFilterAndLinearRegression {

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
      System.err.println("\nUsage: PLSFilterAndLinearRegression <dataset> [classindex]\n");
      System.exit(1);
    }

    // load data
    Instances data = ExampleHelper.loadData(args[0], (args.length > 1) ? args[1] : null);

    // configure classifier we want to optimize
    PLSFilter pls = new PLSFilter();
    LinearRegression lr = new LinearRegression();
    FilteredClassifier fc = new FilteredClassifier();
    fc.setClassifier(lr);
    fc.setFilter(pls);
    fc.setDoNotCheckForModifiedClassAttribute(true);

    // configure multisearch
    // 1. number of components
    ListParameter numComp = new ListParameter();
    numComp.setProperty("filter.numComponents");
    numComp.setList("2 5 7");
    // 2. ridge
    MathParameter ridge = new MathParameter();
    ridge.setProperty("classifier.ridge");
    ridge.setBase(10);
    ridge.setMin(-5);
    ridge.setMax(1);
    ridge.setStep(1);
    ridge.setExpression("pow(BASE,I)");
    // assemble everything
    MultiSearch multi = new MultiSearch();
    multi.setClassifier(fc);
    multi.setSearchParameters(new AbstractParameter[]{
      numComp,
      ridge
    });

    // output configuration
    System.out.println("\nMultiSearch commandline:\n" + Utils.toCommandLine(multi));

    // optimize
    System.out.println("\nOptimizing...\n");
    multi.buildClassifier(data);
    System.out.println("Best setup:\n" + Utils.toCommandLine(multi.getBestClassifier()));
  }
}
