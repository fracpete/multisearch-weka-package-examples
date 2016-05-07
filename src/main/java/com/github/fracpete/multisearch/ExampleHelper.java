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
 * MultiSearchHelper.java
 * Copyright (C) 2016 University of Waikato, Hamilton, NZ
 */

package com.github.fracpete.multisearch;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Helper class for the examples.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class ExampleHelper {

  /**
   * Loads the dataset from disk.
   *
   * @param filename	the file to load
   * @param classIndex	the 1-based class index (first and last accepted as well),
   *                    uses last attribute if null
   * @return		the dataset
   * @throws Exception	if loading of data fails
   */
  public static Instances loadData(String filename, String classIndex) throws Exception {
    Instances data = DataSource.read(filename);
    if (classIndex != null) {
      if (classIndex.equals("first"))
	data.setClassIndex(0);
      else if (classIndex.equals("last"))
	data.setClassIndex(data.numAttributes() - 1);
      else
	data.setClassIndex(Integer.parseInt(classIndex) - 1);
    }
    else {
      data.setClassIndex(data.numAttributes() - 1);
    }
    return data;
  }
}
