/**
 * Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version
 * 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package org.apache.storm.streams.processors;

import java.util.HashMap;

import org.apache.storm.streams.StreamMonitor;
import org.apache.storm.streams.operations.Predicate;

public class FilterProcessor<T> extends BaseProcessor<T> {
    private final Predicate<T> predicate;

    public FilterProcessor(Predicate<T> predicate) {
        this(predicate, new HashMap<>());
    }

    public FilterProcessor(Predicate<T> predicate, HashMap<String, Object> description) {
        this.predicate = predicate;
        this.description = description;
        this.monitor = new StreamMonitor(description, this);
    }

    @Override
    public void execute(T input) {
        monitor.reportInput(input);
        if (predicate.test(input)) {
            monitor.reportOutput(input);
            context.forward(input);
        }
    }

}
