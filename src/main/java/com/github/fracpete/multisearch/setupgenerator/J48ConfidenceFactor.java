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

package com.github.fracpete.multisearch.setupgenerator;

import weka.classifiers.meta.MultiSearch;
import weka.classifiers.trees.J48;
import weka.core.SetupGenerator;
import weka.core.Utils;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.MathParameter;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * Varies the confidence factor of the J48 classifier.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class J48ConfidenceFactor {

  /**
   * Outputs the commandlines.
   *
   * @param args	the commandline options
   * @throws Exception	if optimization fails for some reason
   */
  public static void main(String[] args) throws Exception {
    // configure classifier we want to optimize
    J48 j48 = new J48();

    // configure generator
    MathParameter conf = new MathParameter();
    conf.setProperty("confidenceFactor");
    conf.setBase(10);
    conf.setMin(0.05);
    conf.setMax(0.75);
    conf.setStep(0.05);
    conf.setExpression("I");
    MultiSearch multi = new MultiSearch();
    multi.setClassifier(j48);
    SetupGenerator generator = new SetupGenerator();
    generator.setBaseObject(j48);
    generator.setParameters(new AbstractParameter[]{
      conf
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
