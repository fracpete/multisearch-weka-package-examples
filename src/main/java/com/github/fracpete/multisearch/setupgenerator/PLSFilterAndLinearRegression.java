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

package com.github.fracpete.multisearch.setupgenerator;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.SetupGenerator;
import weka.core.Utils;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.ListParameter;
import weka.core.setupgenerator.MathParameter;
import weka.filters.supervised.attribute.PLSFilter;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * Generates commandline setups varying the number of components of PLSFilter
 * and the ridge parameter of LinearRegression.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class PLSFilterAndLinearRegression {

  /**
   * Outputs the commandlines.
   *
   * @param args	the commandline options
   * @throws Exception	if setup generator fails for some reason
   */
  public static void main(String[] args) throws Exception {
    // configure classifier we want to generate setups for
    PLSFilter pls = new PLSFilter();
    LinearRegression lr = new LinearRegression();
    FilteredClassifier fc = new FilteredClassifier();
    fc.setClassifier(lr);
    fc.setFilter(pls);
    // required for Weka > 3.7.13
    fc.setDoNotCheckForModifiedClassAttribute(true);

    // configure generator
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
    SetupGenerator generator = new SetupGenerator();
    generator.setBaseObject(fc);
    generator.setParameters(new AbstractParameter[]{
      numComp,
      ridge
    });

    // output configuration
    System.out.println("\nSetupgenerator commandline:\n" + Utils.toCommandLine(generator));

    // output commandlines
    System.out.println("\nCommandlines:\n");
    Enumeration<Serializable> enm = generator.setups();
    while (enm.hasMoreElements())
      System.out.println(Utils.toCommandLine(enm.nextElement()));
  }
}
