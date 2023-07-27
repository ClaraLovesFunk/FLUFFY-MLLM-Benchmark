# Testing-Multimodal-LLMs

## Benchmark



### run1
<table id="T_ed6a7">
<thead>
<tr>
<th class="index_name level0">Dataset</th>
<th class="col_heading level0 col0" colspan="2" id="T_ed6a7_level0_col0" style="border-bottom: 1px solid black;">aokvqa</th>
<th class="col_heading level0 col2" id="T_ed6a7_level0_col2" style="border-bottom: 1px solid black;">clevr</th>
<th class="col_heading level0 col3" colspan="4" id="T_ed6a7_level0_col3" style="border-bottom: 1px solid black;">esnlive</th>
<th class="col_heading level0 col7" id="T_ed6a7_level0_col7" style="border-bottom: 1px solid black;">gqa</th>
<th class="col_heading level0 col8" colspan="4" id="T_ed6a7_level0_col8" style="border-bottom: 1px solid black;">hateful_memes</th>
<th class="col_heading level0 col12" colspan="4" id="T_ed6a7_level0_col12" style="border-bottom: 1px solid black;">mami</th>
<th class="col_heading level0 col16" colspan="4" id="T_ed6a7_level0_col16" style="border-bottom: 1px solid black;">mvsa</th>
<th class="col_heading level0 col20" id="T_ed6a7_level0_col20" style="border-bottom: 1px solid black;">okvqa</th>
<th class="col_heading level0 col21" id="T_ed6a7_level0_col21" style="border-bottom: 1px solid black;">scienceqa</th>
</tr>
<tr>
<th class="index_name level1">Task</th>
<th class="col_heading level1 col0" id="T_ed6a7_level1_col0">direct answer</th>
<th class="col_heading level1 col1" id="T_ed6a7_level1_col1">multiple choice</th>
<th class="col_heading level1 col2" id="T_ed6a7_level1_col2">direct answer</th>
<th class="col_heading level1 col3" colspan="4" id="T_ed6a7_level1_col3">entailment prediction</th>
<th class="col_heading level1 col7" id="T_ed6a7_level1_col7">direct answer</th>
<th class="col_heading level1 col8" colspan="4" id="T_ed6a7_level1_col8">hate classification</th>
<th class="col_heading level1 col12" colspan="4" id="T_ed6a7_level1_col12">sexism classification</th>
<th class="col_heading level1 col16" colspan="4" id="T_ed6a7_level1_col16">sentiment analysis</th>
<th class="col_heading level1 col20" id="T_ed6a7_level1_col20">direct answer</th>
<th class="col_heading level1 col21" id="T_ed6a7_level1_col21">multiple choice (sqa)</th>
</tr>
<tr>
<th class="index_name level2">Metric</th>
<th class="col_heading level2 col0" id="T_ed6a7_level2_col0">accuracy</th>
<th class="col_heading level2 col1" id="T_ed6a7_level2_col1">accuracy</th>
<th class="col_heading level2 col2" id="T_ed6a7_level2_col2">accuracy</th>
<th class="col_heading level2 col3" id="T_ed6a7_level2_col3">accuracy</th>
<th class="col_heading level2 col4" id="T_ed6a7_level2_col4">f1 (weighted)</th>
<th class="col_heading level2 col5" id="T_ed6a7_level2_col5">precision (weighted)</th>
<th class="col_heading level2 col6" id="T_ed6a7_level2_col6">recall (weighted)</th>
<th class="col_heading level2 col7" id="T_ed6a7_level2_col7">accuracy</th>
<th class="col_heading level2 col8" id="T_ed6a7_level2_col8">accuracy</th>
<th class="col_heading level2 col9" id="T_ed6a7_level2_col9">f1</th>
<th class="col_heading level2 col10" id="T_ed6a7_level2_col10">precision</th>
<th class="col_heading level2 col11" id="T_ed6a7_level2_col11">recall</th>
<th class="col_heading level2 col12" id="T_ed6a7_level2_col12">accuracy</th>
<th class="col_heading level2 col13" id="T_ed6a7_level2_col13">f1 (weighted)</th>
<th class="col_heading level2 col14" id="T_ed6a7_level2_col14">precision (weighted)</th>
<th class="col_heading level2 col15" id="T_ed6a7_level2_col15">recall (weighted)</th>
<th class="col_heading level2 col16" id="T_ed6a7_level2_col16">accuracy</th>
<th class="col_heading level2 col17" id="T_ed6a7_level2_col17">f1 (weighted)</th>
<th class="col_heading level2 col18" id="T_ed6a7_level2_col18">precision (weighted)</th>
<th class="col_heading level2 col19" id="T_ed6a7_level2_col19">recall (weighted)</th>
<th class="col_heading level2 col20" id="T_ed6a7_level2_col20">accuracy</th>
<th class="col_heading level2 col21" id="T_ed6a7_level2_col21">accuracy</th>
</tr>
<tr>
<th class="index_name level0"></th>
<th class="blank col0"> </th>
<th class="blank col1"> </th>
<th class="blank col2"> </th>
<th class="blank col3"> </th>
<th class="blank col4"> </th>
<th class="blank col5"> </th>
<th class="blank col6"> </th>
<th class="blank col7"> </th>
<th class="blank col8"> </th>
<th class="blank col9"> </th>
<th class="blank col10"> </th>
<th class="blank col11"> </th>
<th class="blank col12"> </th>
<th class="blank col13"> </th>
<th class="blank col14"> </th>
<th class="blank col15"> </th>
<th class="blank col16"> </th>
<th class="blank col17"> </th>
<th class="blank col18"> </th>
<th class="blank col19"> </th>
<th class="blank col20"> </th>
<th class="blank col21"> </th>
</tr>
</thead>
<tbody>
<tr>
<th class="row_heading level0 row0" id="T_ed6a7_level0_row0">blip2</th>
<td class="data row0 col0" id="T_ed6a7_row0_col0">0.24</td>
<td class="data row0 col1" id="T_ed6a7_row0_col1">0.70</td>
<td class="data row0 col2" id="T_ed6a7_row0_col2">0.26</td>
<td class="data row0 col3" id="T_ed6a7_row0_col3">0.54</td>
<td class="data row0 col4" id="T_ed6a7_row0_col4">0.51</td>
<td class="data row0 col5" id="T_ed6a7_row0_col5">0.73</td>
<td class="data row0 col6" id="T_ed6a7_row0_col6">0.54</td>
<td class="data row0 col7" id="T_ed6a7_row0_col7">0.32</td>
<td class="data row0 col8" id="T_ed6a7_row0_col8">0.60</td>
<td class="data row0 col9" id="T_ed6a7_row0_col9">0.56</td>
<td class="data row0 col10" id="T_ed6a7_row0_col10">0.62</td>
<td class="data row0 col11" id="T_ed6a7_row0_col11">0.51</td>
<td class="data row0 col12" id="T_ed6a7_row0_col12">0.60</td>
<td class="data row0 col13" id="T_ed6a7_row0_col13">0.56</td>
<td class="data row0 col14" id="T_ed6a7_row0_col14">0.66</td>
<td class="data row0 col15" id="T_ed6a7_row0_col15">0.60</td>
<td class="data row0 col16" id="T_ed6a7_row0_col16">0.69</td>
<td class="data row0 col17" id="T_ed6a7_row0_col17">0.67</td>
<td class="data row0 col18" id="T_ed6a7_row0_col18">0.67</td>
<td class="data row0 col19" id="T_ed6a7_row0_col19">0.69</td>
<td class="data row0 col20" id="T_ed6a7_row0_col20">0.18</td>
<td class="data row0 col21" id="T_ed6a7_row0_col21">0.36</td>
</tr>
</tbody>
</table>
