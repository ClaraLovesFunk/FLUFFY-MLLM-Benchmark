<img src="utils_general/fluffy.png" width="100%" />

## Benchmark



### Metrics - Evaluation with Post-Processing Tolerance


<table id="T_60762">
<thead>
<tr>
<th class="index_name level0">Dataset</th>
<th class="col_heading level0 col0" colspan="2" id="T_60762_level0_col0" style="border-bottom: 1px solid black;">aokvqa</th>
<th class="col_heading level0 col2" id="T_60762_level0_col2" style="border-bottom: 1px solid black;">clevr</th>
<th class="col_heading level0 col3" colspan="4" id="T_60762_level0_col3" style="border-bottom: 1px solid black;">esnlive</th>
<th class="col_heading level0 col7" id="T_60762_level0_col7" style="border-bottom: 1px solid black;">gqa</th>
<th class="col_heading level0 col8" colspan="4" id="T_60762_level0_col8" style="border-bottom: 1px solid black;">hateful_memes</th>
<th class="col_heading level0 col12" colspan="4" id="T_60762_level0_col12" style="border-bottom: 1px solid black;">mami</th>
<th class="col_heading level0 col16" colspan="4" id="T_60762_level0_col16" style="border-bottom: 1px solid black;">mvsa</th>
<th class="col_heading level0 col20" id="T_60762_level0_col20" style="border-bottom: 1px solid black;">okvqa</th>
<th class="col_heading level0 col21" id="T_60762_level0_col21" style="border-bottom: 1px solid black;">scienceqa</th>
<th class="col_heading level0 col22" id="T_60762_level0_col22" style="border-bottom: 1px solid black;">Average Accuracy</th>
</tr>
<tr>
<th class="index_name level1">Task</th>
<th class="col_heading level1 col0" id="T_60762_level1_col0">direct answer (aokvqa)</th>
<th class="col_heading level1 col1" id="T_60762_level1_col1">multiple choice (aokvqa)</th>
<th class="col_heading level1 col2" id="T_60762_level1_col2">direct answer (clevr)</th>
<th class="col_heading level1 col3" colspan="4" id="T_60762_level1_col3">entailment prediction</th>
<th class="col_heading level1 col7" id="T_60762_level1_col7">direct answer (gqa)</th>
<th class="col_heading level1 col8" colspan="4" id="T_60762_level1_col8">hate classification</th>
<th class="col_heading level1 col12" colspan="4" id="T_60762_level1_col12">sexism classification</th>
<th class="col_heading level1 col16" colspan="4" id="T_60762_level1_col16">sentiment analysis</th>
<th class="col_heading level1 col20" id="T_60762_level1_col20">direct answer (okvqa)</th>
<th class="col_heading level1 col21" id="T_60762_level1_col21">multiple choice (sqa)</th>
<th class="col_heading level1 col22" id="T_60762_level1_col22"></th>
</tr>
<tr>
<th class="index_name level2">Metric</th>
<th class="col_heading level2 col0" id="T_60762_level2_col0">accuracy</th>
<th class="col_heading level2 col1" id="T_60762_level2_col1">accuracy</th>
<th class="col_heading level2 col2" id="T_60762_level2_col2">accuracy</th>
<th class="col_heading level2 col3" id="T_60762_level2_col3">accuracy</th>
<th class="col_heading level2 col4" id="T_60762_level2_col4">f1</th>
<th class="col_heading level2 col5" id="T_60762_level2_col5">precision</th>
<th class="col_heading level2 col6" id="T_60762_level2_col6">recall</th>
<th class="col_heading level2 col7" id="T_60762_level2_col7">accuracy</th>
<th class="col_heading level2 col8" id="T_60762_level2_col8">accuracy</th>
<th class="col_heading level2 col9" id="T_60762_level2_col9">f1</th>
<th class="col_heading level2 col10" id="T_60762_level2_col10">precision</th>
<th class="col_heading level2 col11" id="T_60762_level2_col11">recall</th>
<th class="col_heading level2 col12" id="T_60762_level2_col12">accuracy</th>
<th class="col_heading level2 col13" id="T_60762_level2_col13">f1</th>
<th class="col_heading level2 col14" id="T_60762_level2_col14">precision</th>
<th class="col_heading level2 col15" id="T_60762_level2_col15">recall</th>
<th class="col_heading level2 col16" id="T_60762_level2_col16">accuracy</th>
<th class="col_heading level2 col17" id="T_60762_level2_col17">f1</th>
<th class="col_heading level2 col18" id="T_60762_level2_col18">precision</th>
<th class="col_heading level2 col19" id="T_60762_level2_col19">recall</th>
<th class="col_heading level2 col20" id="T_60762_level2_col20">accuracy</th>
<th class="col_heading level2 col21" id="T_60762_level2_col21">accuracy</th>
<th class="col_heading level2 col22" id="T_60762_level2_col22"></th>
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
<th class="blank col22"> </th>
</tr>
</thead>
<tbody>
<tr>
<th class="row_heading level0 row0" id="T_60762_level0_row0">adept</th>
<td class="data row0 col0" id="T_60762_row0_col0">0.50</td>
<td class="data row0 col1" id="T_60762_row0_col1">0.37</td>
<td class="data row0 col2" id="T_60762_row0_col2">0.34</td>
<td class="data row0 col3" id="T_60762_row0_col3">0.33</td>
<td class="data row0 col4" id="T_60762_row0_col4">0.25</td>
<td class="data row0 col5" id="T_60762_row0_col5">0.61</td>
<td class="data row0 col6" id="T_60762_row0_col6">0.33</td>
<td class="data row0 col7" id="T_60762_row0_col7">0.44</td>
<td class="data row0 col8" id="T_60762_row0_col8">0.55</td>
<td class="data row0 col9" id="T_60762_row0_col9">0.63</td>
<td class="data row0 col10" id="T_60762_row0_col10">0.60</td>
<td class="data row0 col11" id="T_60762_row0_col11">0.67</td>
<td class="data row0 col12" id="T_60762_row0_col12">0.53</td>
<td class="data row0 col13" id="T_60762_row0_col13">0.64</td>
<td class="data row0 col14" id="T_60762_row0_col14">0.55</td>
<td class="data row0 col15" id="T_60762_row0_col15">0.75</td>
<td class="data row0 col16" id="T_60762_row0_col16">0.59</td>
<td class="data row0 col17" id="T_60762_row0_col17">0.54</td>
<td class="data row0 col18" id="T_60762_row0_col18">0.58</td>
<td class="data row0 col19" id="T_60762_row0_col19">0.59</td>
<td class="data row0 col20" id="T_60762_row0_col20">0.48</td>
<td class="data row0 col21" id="T_60762_row0_col21">0.32</td>
<td class="data row0 col22" id="T_60762_row0_col22">0.44</td>
</tr>
<tr>
<th class="row_heading level0 row1" id="T_60762_level0_row1">blip2</th>
<td class="data row1 col0" id="T_60762_row1_col0">0.44</td>
<td class="data row1 col1" id="T_60762_row1_col1">0.69</td>
<td class="data row1 col2" id="T_60762_row1_col2">0.28</td>
<td class="data row1 col3" id="T_60762_row1_col3">0.54</td>
<td class="data row1 col4" id="T_60762_row1_col4">0.51</td>
<td class="data row1 col5" id="T_60762_row1_col5">0.72</td>
<td class="data row1 col6" id="T_60762_row1_col6">0.54</td>
<td class="data row1 col7" id="T_60762_row1_col7">0.33</td>
<td class="data row1 col8" id="T_60762_row1_col8">0.61</td>
<td class="data row1 col9" id="T_60762_row1_col9">0.61</td>
<td class="data row1 col10" id="T_60762_row1_col10">0.60</td>
<td class="data row1 col11" id="T_60762_row1_col11">0.61</td>
<td class="data row1 col12" id="T_60762_row1_col12">0.52</td>
<td class="data row1 col13" id="T_60762_row1_col13">0.67</td>
<td class="data row1 col14" id="T_60762_row1_col14">0.51</td>
<td class="data row1 col15" id="T_60762_row1_col15">0.99</td>
<td class="data row1 col16" id="T_60762_row1_col16">0.68</td>
<td class="data row1 col17" id="T_60762_row1_col17">0.68</td>
<td class="data row1 col18" id="T_60762_row1_col18">0.68</td>
<td class="data row1 col19" id="T_60762_row1_col19">0.68</td>
<td class="data row1 col20" id="T_60762_row1_col20">0.35</td>
<td class="data row1 col21" id="T_60762_row1_col21">0.37</td>
<td class="data row1 col22" id="T_60762_row1_col22">0.47</td>
</tr>
<tr>
<th class="row_heading level0 row2" id="T_60762_level0_row2">idefics</th>
<td class="data row2 col0" id="T_60762_row2_col0">0.17</td>
<td class="data row2 col1" id="T_60762_row2_col1">0.30</td>
<td class="data row2 col2" id="T_60762_row2_col2">0.38</td>
<td class="data row2 col3" id="T_60762_row2_col3">0.42</td>
<td class="data row2 col4" id="T_60762_row2_col4">0.32</td>
<td class="data row2 col5" id="T_60762_row2_col5">0.38</td>
<td class="data row2 col6" id="T_60762_row2_col6">0.42</td>
<td class="data row2 col7" id="T_60762_row2_col7">0.34</td>
<td class="data row2 col8" id="T_60762_row2_col8">0.50</td>
<td class="data row2 col9" id="T_60762_row2_col9">0.67</td>
<td class="data row2 col10" id="T_60762_row2_col10">0.50</td>
<td class="data row2 col11" id="T_60762_row2_col11">1.00</td>
<td class="data row2 col12" id="T_60762_row2_col12">0.52</td>
<td class="data row2 col13" id="T_60762_row2_col13">0.34</td>
<td class="data row2 col14" id="T_60762_row2_col14">0.54</td>
<td class="data row2 col15" id="T_60762_row2_col15">0.24</td>
<td class="data row2 col16" id="T_60762_row2_col16">0.17</td>
<td class="data row2 col17" id="T_60762_row2_col17">0.14</td>
<td class="data row2 col18" id="T_60762_row2_col18">0.73</td>
<td class="data row2 col19" id="T_60762_row2_col19">0.17</td>
<td class="data row2 col20" id="T_60762_row2_col20">0.20</td>
<td class="data row2 col21" id="T_60762_row2_col21">0.32</td>
<td class="data row2 col22" id="T_60762_row2_col22">0.34</td>
</tr>
<tr>
<th class="row_heading level0 row3" id="T_60762_level0_row3">instructblip</th>
<td class="data row3 col0" id="T_60762_row3_col0">0.55</td>
<td class="data row3 col1" id="T_60762_row3_col1">0.74</td>
<td class="data row3 col2" id="T_60762_row3_col2">0.34</td>
<td class="data row3 col3" id="T_60762_row3_col3">0.63</td>
<td class="data row3 col4" id="T_60762_row3_col4">0.64</td>
<td class="data row3 col5" id="T_60762_row3_col5">0.75</td>
<td class="data row3 col6" id="T_60762_row3_col6">0.63</td>
<td class="data row3 col7" id="T_60762_row3_col7">0.50</td>
<td class="data row3 col8" id="T_60762_row3_col8">0.58</td>
<td class="data row3 col9" id="T_60762_row3_col9">0.49</td>
<td class="data row3 col10" id="T_60762_row3_col10">0.63</td>
<td class="data row3 col11" id="T_60762_row3_col11">0.40</td>
<td class="data row3 col12" id="T_60762_row3_col12">0.56</td>
<td class="data row3 col13" id="T_60762_row3_col13">0.57</td>
<td class="data row3 col14" id="T_60762_row3_col14">0.54</td>
<td class="data row3 col15" id="T_60762_row3_col15">0.61</td>
<td class="data row3 col16" id="T_60762_row3_col16">0.68</td>
<td class="data row3 col17" id="T_60762_row3_col17">0.67</td>
<td class="data row3 col18" id="T_60762_row3_col18">0.67</td>
<td class="data row3 col19" id="T_60762_row3_col19">0.68</td>
<td class="data row3 col20" id="T_60762_row3_col20">0.52</td>
<td class="data row3 col21" id="T_60762_row3_col21">0.49</td>
<td class="data row3 col22" id="T_60762_row3_col22">0.55</td>
</tr>
<tr>
<th class="row_heading level0 row4" id="T_60762_level0_row4">llava</th>
<td class="data row4 col0" id="T_60762_row4_col0">0.60</td>
<td class="data row4 col1" id="T_60762_row4_col1">0.62</td>
<td class="data row4 col2" id="T_60762_row4_col2">0.33</td>
<td class="data row4 col3" id="T_60762_row4_col3">0.50</td>
<td class="data row4 col4" id="T_60762_row4_col4">0.46</td>
<td class="data row4 col5" id="T_60762_row4_col5">0.65</td>
<td class="data row4 col6" id="T_60762_row4_col6">0.50</td>
<td class="data row4 col7" id="T_60762_row4_col7">0.51</td>
<td class="data row4 col8" id="T_60762_row4_col8">0.63</td>
<td class="data row4 col9" id="T_60762_row4_col9">0.77</td>
<td class="data row4 col10" id="T_60762_row4_col10">0.63</td>
<td class="data row4 col11" id="T_60762_row4_col11">1.00</td>
<td class="data row4 col12" id="T_60762_row4_col12">0.68</td>
<td class="data row4 col13" id="T_60762_row4_col13">0.81</td>
<td class="data row4 col14" id="T_60762_row4_col14">0.68</td>
<td class="data row4 col15" id="T_60762_row4_col15">1.00</td>
<td class="data row4 col16" id="T_60762_row4_col16">0.68</td>
<td class="data row4 col17" id="T_60762_row4_col17">0.67</td>
<td class="data row4 col18" id="T_60762_row4_col18">0.66</td>
<td class="data row4 col19" id="T_60762_row4_col19">0.68</td>
<td class="data row4 col20" id="T_60762_row4_col20">0.58</td>
<td class="data row4 col21" id="T_60762_row4_col21">0.37</td>
<td class="data row4 col22" id="T_60762_row4_col22">0.54</td>
</tr>
<tr>
<th class="row_heading level0 row5" id="T_60762_level0_row5">openflamingo</th>
<td class="data row5 col0" id="T_60762_row5_col0">0.37</td>
<td class="data row5 col1" id="T_60762_row5_col1">0.24</td>
<td class="data row5 col2" id="T_60762_row5_col2">0.22</td>
<td class="data row5 col3" id="T_60762_row5_col3">0.38</td>
<td class="data row5 col4" id="T_60762_row5_col4">0.32</td>
<td class="data row5 col5" id="T_60762_row5_col5">0.38</td>
<td class="data row5 col6" id="T_60762_row5_col6">0.38</td>
<td class="data row5 col7" id="T_60762_row5_col7">0.34</td>
<td class="data row5 col8" id="T_60762_row5_col8">0.46</td>
<td class="data row5 col9" id="T_60762_row5_col9">0.49</td>
<td class="data row5 col10" id="T_60762_row5_col10">0.47</td>
<td class="data row5 col11" id="T_60762_row5_col11">0.50</td>
<td class="data row5 col12" id="T_60762_row5_col12">0.50</td>
<td class="data row5 col13" id="T_60762_row5_col13">0.20</td>
<td class="data row5 col14" id="T_60762_row5_col14">0.57</td>
<td class="data row5 col15" id="T_60762_row5_col15">0.12</td>
<td class="data row5 col16" id="T_60762_row5_col16">0.57</td>
<td class="data row5 col17" id="T_60762_row5_col17">0.42</td>
<td class="data row5 col18" id="T_60762_row5_col18">0.66</td>
<td class="data row5 col19" id="T_60762_row5_col19">0.57</td>
<td class="data row5 col20" id="T_60762_row5_col20">0.38</td>
<td class="data row5 col21" id="T_60762_row5_col21">0.38</td>
<td class="data row5 col22" id="T_60762_row5_col22">0.39</td>
</tr>
<tr>
<th class="row_heading level0 row6" id="T_60762_level0_row6">otter</th>
<td class="data row6 col0" id="T_60762_row6_col0">0.37</td>
<td class="data row6 col1" id="T_60762_row6_col1">0.45</td>
<td class="data row6 col2" id="T_60762_row6_col2">0.23</td>
<td class="data row6 col3" id="T_60762_row6_col3">0.38</td>
<td class="data row6 col4" id="T_60762_row6_col4">0.32</td>
<td class="data row6 col5" id="T_60762_row6_col5">0.42</td>
<td class="data row6 col6" id="T_60762_row6_col6">0.38</td>
<td class="data row6 col7" id="T_60762_row6_col7">0.38</td>
<td class="data row6 col8" id="T_60762_row6_col8">0.51</td>
<td class="data row6 col9" id="T_60762_row6_col9">0.19</td>
<td class="data row6 col10" id="T_60762_row6_col10">0.54</td>
<td class="data row6 col11" id="T_60762_row6_col11">0.11</td>
<td class="data row6 col12" id="T_60762_row6_col12">0.57</td>
<td class="data row6 col13" id="T_60762_row6_col13">0.43</td>
<td class="data row6 col14" id="T_60762_row6_col14">0.65</td>
<td class="data row6 col15" id="T_60762_row6_col15">0.32</td>
<td class="data row6 col16" id="T_60762_row6_col16">0.59</td>
<td class="data row6 col17" id="T_60762_row6_col17">0.60</td>
<td class="data row6 col18" id="T_60762_row6_col18">0.65</td>
<td class="data row6 col19" id="T_60762_row6_col19">0.59</td>
<td class="data row6 col20" id="T_60762_row6_col20">0.42</td>
<td class="data row6 col21" id="T_60762_row6_col21">0.39</td>
<td class="data row6 col22" id="T_60762_row6_col22">0.43</td>
</tr>
</tbody>
</table>


### Valid Answer Ratio - Evaluation with Post-Processing Tolerance


<table id="T_23cf9">
<thead>
<tr>
<th class="index_name level0">Dataset</th>
<th class="col_heading level0 col0" colspan="2" id="T_23cf9_level0_col0" style="border-bottom: 1px solid black;">aokvqa</th>
<th class="col_heading level0 col2" id="T_23cf9_level0_col2" style="border-bottom: 1px solid black;">clevr</th>
<th class="col_heading level0 col3" id="T_23cf9_level0_col3" style="border-bottom: 1px solid black;">esnlive</th>
<th class="col_heading level0 col4" id="T_23cf9_level0_col4" style="border-bottom: 1px solid black;">gqa</th>
<th class="col_heading level0 col5" id="T_23cf9_level0_col5" style="border-bottom: 1px solid black;">hateful_memes</th>
<th class="col_heading level0 col6" id="T_23cf9_level0_col6" style="border-bottom: 1px solid black;">mami</th>
<th class="col_heading level0 col7" id="T_23cf9_level0_col7" style="border-bottom: 1px solid black;">mvsa</th>
<th class="col_heading level0 col8" id="T_23cf9_level0_col8" style="border-bottom: 1px solid black;">okvqa</th>
<th class="col_heading level0 col9" id="T_23cf9_level0_col9" style="border-bottom: 1px solid black;">scienceqa</th>
<th class="col_heading level0 col10" id="T_23cf9_level0_col10" style="border-bottom: 1px solid black;">Average Ratio</th>
</tr>
<tr>
<th class="index_name level1">Task</th>
<th class="col_heading level1 col0" id="T_23cf9_level1_col0">direct answer (aokvqa)</th>
<th class="col_heading level1 col1" id="T_23cf9_level1_col1">multiple choice (aokvqa)</th>
<th class="col_heading level1 col2" id="T_23cf9_level1_col2">direct answer (clevr)</th>
<th class="col_heading level1 col3" id="T_23cf9_level1_col3">entailment prediction</th>
<th class="col_heading level1 col4" id="T_23cf9_level1_col4">direct answer (gqa)</th>
<th class="col_heading level1 col5" id="T_23cf9_level1_col5">hate classification</th>
<th class="col_heading level1 col6" id="T_23cf9_level1_col6">sexism classification</th>
<th class="col_heading level1 col7" id="T_23cf9_level1_col7">sentiment analysis</th>
<th class="col_heading level1 col8" id="T_23cf9_level1_col8">direct answer (okvqa)</th>
<th class="col_heading level1 col9" id="T_23cf9_level1_col9">multiple choice (sqa)</th>
<th class="col_heading level1 col10" id="T_23cf9_level1_col10"></th>
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
</tr>
</thead>
<tbody>
<tr>
<th class="row_heading level0 row0" id="T_23cf9_level0_row0">adept</th>
<td class="data row0 col0" id="T_23cf9_row0_col0">1.00</td>
<td class="data row0 col1" id="T_23cf9_row0_col1">0.86</td>
<td class="data row0 col2" id="T_23cf9_row0_col2">1.00</td>
<td class="data row0 col3" id="T_23cf9_row0_col3">0.99</td>
<td class="data row0 col4" id="T_23cf9_row0_col4">1.00</td>
<td class="data row0 col5" id="T_23cf9_row0_col5">0.13</td>
<td class="data row0 col6" id="T_23cf9_row0_col6">0.26</td>
<td class="data row0 col7" id="T_23cf9_row0_col7">0.99</td>
<td class="data row0 col8" id="T_23cf9_row0_col8">1.00</td>
<td class="data row0 col9" id="T_23cf9_row0_col9">0.06</td>
<td class="data row0 col10" id="T_23cf9_row0_col10">0.73</td>
</tr>
<tr>
<th class="row_heading level0 row1" id="T_23cf9_level0_row1">blip2</th>
<td class="data row1 col0" id="T_23cf9_row1_col0">1.00</td>
<td class="data row1 col1" id="T_23cf9_row1_col1">1.00</td>
<td class="data row1 col2" id="T_23cf9_row1_col2">1.00</td>
<td class="data row1 col3" id="T_23cf9_row1_col3">1.00</td>
<td class="data row1 col4" id="T_23cf9_row1_col4">1.00</td>
<td class="data row1 col5" id="T_23cf9_row1_col5">1.00</td>
<td class="data row1 col6" id="T_23cf9_row1_col6">1.00</td>
<td class="data row1 col7" id="T_23cf9_row1_col7">1.00</td>
<td class="data row1 col8" id="T_23cf9_row1_col8">1.00</td>
<td class="data row1 col9" id="T_23cf9_row1_col9">0.94</td>
<td class="data row1 col10" id="T_23cf9_row1_col10">0.99</td>
</tr>
<tr>
<th class="row_heading level0 row2" id="T_23cf9_level0_row2">idefics</th>
<td class="data row2 col0" id="T_23cf9_row2_col0">1.00</td>
<td class="data row2 col1" id="T_23cf9_row2_col1">0.85</td>
<td class="data row2 col2" id="T_23cf9_row2_col2">1.00</td>
<td class="data row2 col3" id="T_23cf9_row2_col3">0.31</td>
<td class="data row2 col4" id="T_23cf9_row2_col4">1.00</td>
<td class="data row2 col5" id="T_23cf9_row2_col5">1.00</td>
<td class="data row2 col6" id="T_23cf9_row2_col6">0.94</td>
<td class="data row2 col7" id="T_23cf9_row2_col7">1.00</td>
<td class="data row2 col8" id="T_23cf9_row2_col8">1.00</td>
<td class="data row2 col9" id="T_23cf9_row2_col9">0.05</td>
<td class="data row2 col10" id="T_23cf9_row2_col10">0.81</td>
</tr>
<tr>
<th class="row_heading level0 row3" id="T_23cf9_level0_row3">instructblip</th>
<td class="data row3 col0" id="T_23cf9_row3_col0">1.00</td>
<td class="data row3 col1" id="T_23cf9_row3_col1">1.00</td>
<td class="data row3 col2" id="T_23cf9_row3_col2">1.00</td>
<td class="data row3 col3" id="T_23cf9_row3_col3">1.00</td>
<td class="data row3 col4" id="T_23cf9_row3_col4">1.00</td>
<td class="data row3 col5" id="T_23cf9_row3_col5">1.00</td>
<td class="data row3 col6" id="T_23cf9_row3_col6">0.93</td>
<td class="data row3 col7" id="T_23cf9_row3_col7">1.00</td>
<td class="data row3 col8" id="T_23cf9_row3_col8">1.00</td>
<td class="data row3 col9" id="T_23cf9_row3_col9">0.10</td>
<td class="data row3 col10" id="T_23cf9_row3_col10">0.90</td>
</tr>
<tr>
<th class="row_heading level0 row4" id="T_23cf9_level0_row4">llava</th>
<td class="data row4 col0" id="T_23cf9_row4_col0">1.00</td>
<td class="data row4 col1" id="T_23cf9_row4_col1">0.87</td>
<td class="data row4 col2" id="T_23cf9_row4_col2">1.00</td>
<td class="data row4 col3" id="T_23cf9_row4_col3">0.44</td>
<td class="data row4 col4" id="T_23cf9_row4_col4">1.00</td>
<td class="data row4 col5" id="T_23cf9_row4_col5">0.29</td>
<td class="data row4 col6" id="T_23cf9_row4_col6">0.48</td>
<td class="data row4 col7" id="T_23cf9_row4_col7">0.87</td>
<td class="data row4 col8" id="T_23cf9_row4_col8">1.00</td>
<td class="data row4 col9" id="T_23cf9_row4_col9">0.07</td>
<td class="data row4 col10" id="T_23cf9_row4_col10">0.70</td>
</tr>
<tr>
<th class="row_heading level0 row5" id="T_23cf9_level0_row5">openflamingo</th>
<td class="data row5 col0" id="T_23cf9_row5_col0">1.00</td>
<td class="data row5 col1" id="T_23cf9_row5_col1">0.45</td>
<td class="data row5 col2" id="T_23cf9_row5_col2">1.00</td>
<td class="data row5 col3" id="T_23cf9_row5_col3">0.34</td>
<td class="data row5 col4" id="T_23cf9_row5_col4">1.00</td>
<td class="data row5 col5" id="T_23cf9_row5_col5">0.40</td>
<td class="data row5 col6" id="T_23cf9_row5_col6">0.70</td>
<td class="data row5 col7" id="T_23cf9_row5_col7">0.84</td>
<td class="data row5 col8" id="T_23cf9_row5_col8">1.00</td>
<td class="data row5 col9" id="T_23cf9_row5_col9">0.04</td>
<td class="data row5 col10" id="T_23cf9_row5_col10">0.68</td>
</tr>
<tr>
<th class="row_heading level0 row6" id="T_23cf9_level0_row6">otter</th>
<td class="data row6 col0" id="T_23cf9_row6_col0">1.00</td>
<td class="data row6 col1" id="T_23cf9_row6_col1">0.78</td>
<td class="data row6 col2" id="T_23cf9_row6_col2">1.00</td>
<td class="data row6 col3" id="T_23cf9_row6_col3">0.63</td>
<td class="data row6 col4" id="T_23cf9_row6_col4">1.00</td>
<td class="data row6 col5" id="T_23cf9_row6_col5">0.92</td>
<td class="data row6 col6" id="T_23cf9_row6_col6">0.76</td>
<td class="data row6 col7" id="T_23cf9_row6_col7">1.00</td>
<td class="data row6 col8" id="T_23cf9_row6_col8">1.00</td>
<td class="data row6 col9" id="T_23cf9_row6_col9">0.09</td>
<td class="data row6 col10" id="T_23cf9_row6_col10">0.82</td>
</tr>
</tbody>
</table>


### Metrics - Evaluation without Post-Processing Tolerance


<table id="T_00aa2">
<thead>
<tr>
<th class="index_name level0">Dataset</th>
<th class="col_heading level0 col0" colspan="2" id="T_00aa2_level0_col0" style="border-bottom: 1px solid black;">aokvqa</th>
<th class="col_heading level0 col2" id="T_00aa2_level0_col2" style="border-bottom: 1px solid black;">clevr</th>
<th class="col_heading level0 col3" colspan="4" id="T_00aa2_level0_col3" style="border-bottom: 1px solid black;">esnlive</th>
<th class="col_heading level0 col7" id="T_00aa2_level0_col7" style="border-bottom: 1px solid black;">gqa</th>
<th class="col_heading level0 col8" colspan="4" id="T_00aa2_level0_col8" style="border-bottom: 1px solid black;">hateful_memes</th>
<th class="col_heading level0 col12" colspan="4" id="T_00aa2_level0_col12" style="border-bottom: 1px solid black;">mami</th>
<th class="col_heading level0 col16" colspan="4" id="T_00aa2_level0_col16" style="border-bottom: 1px solid black;">mvsa</th>
<th class="col_heading level0 col20" id="T_00aa2_level0_col20" style="border-bottom: 1px solid black;">okvqa</th>
<th class="col_heading level0 col21" id="T_00aa2_level0_col21" style="border-bottom: 1px solid black;">scienceqa</th>
<th class="col_heading level0 col22" id="T_00aa2_level0_col22" style="border-bottom: 1px solid black;">Average Accuracy</th>
</tr>
<tr>
<th class="index_name level1">Task</th>
<th class="col_heading level1 col0" id="T_00aa2_level1_col0">direct answer (aokvqa)</th>
<th class="col_heading level1 col1" id="T_00aa2_level1_col1">multiple choice (aokvqa)</th>
<th class="col_heading level1 col2" id="T_00aa2_level1_col2">direct answer (clevr)</th>
<th class="col_heading level1 col3" colspan="4" id="T_00aa2_level1_col3">entailment prediction</th>
<th class="col_heading level1 col7" id="T_00aa2_level1_col7">direct answer (gqa)</th>
<th class="col_heading level1 col8" colspan="4" id="T_00aa2_level1_col8">hate classification</th>
<th class="col_heading level1 col12" colspan="4" id="T_00aa2_level1_col12">sexism classification</th>
<th class="col_heading level1 col16" colspan="4" id="T_00aa2_level1_col16">sentiment analysis</th>
<th class="col_heading level1 col20" id="T_00aa2_level1_col20">direct answer (okvqa)</th>
<th class="col_heading level1 col21" id="T_00aa2_level1_col21">multiple choice (sqa)</th>
<th class="col_heading level1 col22" id="T_00aa2_level1_col22"></th>
</tr>
<tr>
<th class="index_name level2">Metric</th>
<th class="col_heading level2 col0" id="T_00aa2_level2_col0">accuracy</th>
<th class="col_heading level2 col1" id="T_00aa2_level2_col1">accuracy</th>
<th class="col_heading level2 col2" id="T_00aa2_level2_col2">accuracy</th>
<th class="col_heading level2 col3" id="T_00aa2_level2_col3">accuracy</th>
<th class="col_heading level2 col4" id="T_00aa2_level2_col4">f1</th>
<th class="col_heading level2 col5" id="T_00aa2_level2_col5">precision</th>
<th class="col_heading level2 col6" id="T_00aa2_level2_col6">recall</th>
<th class="col_heading level2 col7" id="T_00aa2_level2_col7">accuracy</th>
<th class="col_heading level2 col8" id="T_00aa2_level2_col8">accuracy</th>
<th class="col_heading level2 col9" id="T_00aa2_level2_col9">f1</th>
<th class="col_heading level2 col10" id="T_00aa2_level2_col10">precision</th>
<th class="col_heading level2 col11" id="T_00aa2_level2_col11">recall</th>
<th class="col_heading level2 col12" id="T_00aa2_level2_col12">accuracy</th>
<th class="col_heading level2 col13" id="T_00aa2_level2_col13">f1</th>
<th class="col_heading level2 col14" id="T_00aa2_level2_col14">precision</th>
<th class="col_heading level2 col15" id="T_00aa2_level2_col15">recall</th>
<th class="col_heading level2 col16" id="T_00aa2_level2_col16">accuracy</th>
<th class="col_heading level2 col17" id="T_00aa2_level2_col17">f1</th>
<th class="col_heading level2 col18" id="T_00aa2_level2_col18">precision</th>
<th class="col_heading level2 col19" id="T_00aa2_level2_col19">recall</th>
<th class="col_heading level2 col20" id="T_00aa2_level2_col20">accuracy</th>
<th class="col_heading level2 col21" id="T_00aa2_level2_col21">accuracy</th>
<th class="col_heading level2 col22" id="T_00aa2_level2_col22"></th>
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
<th class="blank col22"> </th>
</tr>
</thead>
<tbody>
<tr>
<th class="row_heading level0 row0" id="T_00aa2_level0_row0">adept</th>
<td class="data row0 col0" id="T_00aa2_row0_col0">0.07</td>
<td class="data row0 col1" id="T_00aa2_row0_col1">0.35</td>
<td class="data row0 col2" id="T_00aa2_row0_col2">0.09</td>
<td class="data row0 col3" id="T_00aa2_row0_col3">0.33</td>
<td class="data row0 col4" id="T_00aa2_row0_col4">0.25</td>
<td class="data row0 col5" id="T_00aa2_row0_col5">0.61</td>
<td class="data row0 col6" id="T_00aa2_row0_col6">0.33</td>
<td class="data row0 col7" id="T_00aa2_row0_col7">0.09</td>
<td class="data row0 col8" id="T_00aa2_row0_col8">0.49</td>
<td class="data row0 col9" id="T_00aa2_row0_col9">0.54</td>
<td class="data row0 col10" id="T_00aa2_row0_col10">0.53</td>
<td class="data row0 col11" id="T_00aa2_row0_col11">0.55</td>
<td class="data row0 col12" id="T_00aa2_row0_col12">0.53</td>
<td class="data row0 col13" id="T_00aa2_row0_col13">0.63</td>
<td class="data row0 col14" id="T_00aa2_row0_col14">0.54</td>
<td class="data row0 col15" id="T_00aa2_row0_col15">0.74</td>
<td class="data row0 col16" id="T_00aa2_row0_col16">0.59</td>
<td class="data row0 col17" id="T_00aa2_row0_col17">0.54</td>
<td class="data row0 col18" id="T_00aa2_row0_col18">0.58</td>
<td class="data row0 col19" id="T_00aa2_row0_col19">0.59</td>
<td class="data row0 col20" id="T_00aa2_row0_col20">0.08</td>
<td class="data row0 col21" id="T_00aa2_row0_col21">0.24</td>
<td class="data row0 col22" id="T_00aa2_row0_col22">0.29</td>
</tr>
<tr>
<th class="row_heading level0 row1" id="T_00aa2_level0_row1">blip2</th>
<td class="data row1 col0" id="T_00aa2_row1_col0">0.23</td>
<td class="data row1 col1" id="T_00aa2_row1_col1">0.69</td>
<td class="data row1 col2" id="T_00aa2_row1_col2">0.24</td>
<td class="data row1 col3" id="T_00aa2_row1_col3">0.54</td>
<td class="data row1 col4" id="T_00aa2_row1_col4">0.51</td>
<td class="data row1 col5" id="T_00aa2_row1_col5">0.72</td>
<td class="data row1 col6" id="T_00aa2_row1_col6">0.54</td>
<td class="data row1 col7" id="T_00aa2_row1_col7">0.19</td>
<td class="data row1 col8" id="T_00aa2_row1_col8">0.61</td>
<td class="data row1 col9" id="T_00aa2_row1_col9">0.61</td>
<td class="data row1 col10" id="T_00aa2_row1_col10">0.60</td>
<td class="data row1 col11" id="T_00aa2_row1_col11">0.61</td>
<td class="data row1 col12" id="T_00aa2_row1_col12">0.52</td>
<td class="data row1 col13" id="T_00aa2_row1_col13">0.67</td>
<td class="data row1 col14" id="T_00aa2_row1_col14">0.51</td>
<td class="data row1 col15" id="T_00aa2_row1_col15">0.99</td>
<td class="data row1 col16" id="T_00aa2_row1_col16">0.68</td>
<td class="data row1 col17" id="T_00aa2_row1_col17">0.68</td>
<td class="data row1 col18" id="T_00aa2_row1_col18">0.68</td>
<td class="data row1 col19" id="T_00aa2_row1_col19">0.68</td>
<td class="data row1 col20" id="T_00aa2_row1_col20">0.19</td>
<td class="data row1 col21" id="T_00aa2_row1_col21">0.37</td>
<td class="data row1 col22" id="T_00aa2_row1_col22">0.42</td>
</tr>
<tr>
<th class="row_heading level0 row2" id="T_00aa2_level0_row2">idefics</th>
<td class="data row2 col0" id="T_00aa2_row2_col0">0.00</td>
<td class="data row2 col1" id="T_00aa2_row2_col1">0.02</td>
<td class="data row2 col2" id="T_00aa2_row2_col2">0.00</td>
<td class="data row2 col3" id="T_00aa2_row2_col3">0.37</td>
<td class="data row2 col4" id="T_00aa2_row2_col4">0.20</td>
<td class="data row2 col5" id="T_00aa2_row2_col5">0.46</td>
<td class="data row2 col6" id="T_00aa2_row2_col6">0.37</td>
<td class="data row2 col7" id="T_00aa2_row2_col7">0.00</td>
<td class="data row2 col8" id="T_00aa2_row2_col8">0.50</td>
<td class="data row2 col9" id="T_00aa2_row2_col9">0.67</td>
<td class="data row2 col10" id="T_00aa2_row2_col10">0.50</td>
<td class="data row2 col11" id="T_00aa2_row2_col11">1.00</td>
<td class="data row2 col12" id="T_00aa2_row2_col12">0.52</td>
<td class="data row2 col13" id="T_00aa2_row2_col13">0.34</td>
<td class="data row2 col14" id="T_00aa2_row2_col14">0.54</td>
<td class="data row2 col15" id="T_00aa2_row2_col15">0.24</td>
<td class="data row2 col16" id="T_00aa2_row2_col16">0.00</td>
<td class="data row2 col17" id="T_00aa2_row2_col17">0.00</td>
<td class="data row2 col18" id="T_00aa2_row2_col18">0.00</td>
<td class="data row2 col19" id="T_00aa2_row2_col19">0.00</td>
<td class="data row2 col20" id="T_00aa2_row2_col20">0.00</td>
<td class="data row2 col21" id="T_00aa2_row2_col21">nan</td>
<td class="data row2 col22" id="T_00aa2_row2_col22">0.16</td>
</tr>
<tr>
<th class="row_heading level0 row3" id="T_00aa2_level0_row3">instructblip</th>
<td class="data row3 col0" id="T_00aa2_row3_col0">0.49</td>
<td class="data row3 col1" id="T_00aa2_row3_col1">0.74</td>
<td class="data row3 col2" id="T_00aa2_row3_col2">0.34</td>
<td class="data row3 col3" id="T_00aa2_row3_col3">0.63</td>
<td class="data row3 col4" id="T_00aa2_row3_col4">0.64</td>
<td class="data row3 col5" id="T_00aa2_row3_col5">0.75</td>
<td class="data row3 col6" id="T_00aa2_row3_col6">0.63</td>
<td class="data row3 col7" id="T_00aa2_row3_col7">0.42</td>
<td class="data row3 col8" id="T_00aa2_row3_col8">0.58</td>
<td class="data row3 col9" id="T_00aa2_row3_col9">0.49</td>
<td class="data row3 col10" id="T_00aa2_row3_col10">0.63</td>
<td class="data row3 col11" id="T_00aa2_row3_col11">0.40</td>
<td class="data row3 col12" id="T_00aa2_row3_col12">0.56</td>
<td class="data row3 col13" id="T_00aa2_row3_col13">0.57</td>
<td class="data row3 col14" id="T_00aa2_row3_col14">0.54</td>
<td class="data row3 col15" id="T_00aa2_row3_col15">0.61</td>
<td class="data row3 col16" id="T_00aa2_row3_col16">0.68</td>
<td class="data row3 col17" id="T_00aa2_row3_col17">0.67</td>
<td class="data row3 col18" id="T_00aa2_row3_col18">0.67</td>
<td class="data row3 col19" id="T_00aa2_row3_col19">0.68</td>
<td class="data row3 col20" id="T_00aa2_row3_col20">0.37</td>
<td class="data row3 col21" id="T_00aa2_row3_col21">0.65</td>
<td class="data row3 col22" id="T_00aa2_row3_col22">0.54</td>
</tr>
<tr>
<th class="row_heading level0 row4" id="T_00aa2_level0_row4">llava</th>
<td class="data row4 col0" id="T_00aa2_row4_col0">0.00</td>
<td class="data row4 col1" id="T_00aa2_row4_col1">0.00</td>
<td class="data row4 col2" id="T_00aa2_row4_col2">0.00</td>
<td class="data row4 col3" id="T_00aa2_row4_col3">nan</td>
<td class="data row4 col4" id="T_00aa2_row4_col4">nan</td>
<td class="data row4 col5" id="T_00aa2_row4_col5">nan</td>
<td class="data row4 col6" id="T_00aa2_row4_col6">nan</td>
<td class="data row4 col7" id="T_00aa2_row4_col7">0.00</td>
<td class="data row4 col8" id="T_00aa2_row4_col8">nan</td>
<td class="data row4 col9" id="T_00aa2_row4_col9">nan</td>
<td class="data row4 col10" id="T_00aa2_row4_col10">nan</td>
<td class="data row4 col11" id="T_00aa2_row4_col11">nan</td>
<td class="data row4 col12" id="T_00aa2_row4_col12">nan</td>
<td class="data row4 col13" id="T_00aa2_row4_col13">nan</td>
<td class="data row4 col14" id="T_00aa2_row4_col14">nan</td>
<td class="data row4 col15" id="T_00aa2_row4_col15">nan</td>
<td class="data row4 col16" id="T_00aa2_row4_col16">nan</td>
<td class="data row4 col17" id="T_00aa2_row4_col17">nan</td>
<td class="data row4 col18" id="T_00aa2_row4_col18">nan</td>
<td class="data row4 col19" id="T_00aa2_row4_col19">nan</td>
<td class="data row4 col20" id="T_00aa2_row4_col20">0.00</td>
<td class="data row4 col21" id="T_00aa2_row4_col21">nan</td>
<td class="data row4 col22" id="T_00aa2_row4_col22">0.00</td>
</tr>
<tr>
<th class="row_heading level0 row5" id="T_00aa2_level0_row5">openflamingo</th>
<td class="data row5 col0" id="T_00aa2_row5_col0">0.12</td>
<td class="data row5 col1" id="T_00aa2_row5_col1">0.05</td>
<td class="data row5 col2" id="T_00aa2_row5_col2">0.02</td>
<td class="data row5 col3" id="T_00aa2_row5_col3">0.39</td>
<td class="data row5 col4" id="T_00aa2_row5_col4">0.26</td>
<td class="data row5 col5" id="T_00aa2_row5_col5">0.37</td>
<td class="data row5 col6" id="T_00aa2_row5_col6">0.39</td>
<td class="data row5 col7" id="T_00aa2_row5_col7">0.04</td>
<td class="data row5 col8" id="T_00aa2_row5_col8">0.46</td>
<td class="data row5 col9" id="T_00aa2_row5_col9">0.47</td>
<td class="data row5 col10" id="T_00aa2_row5_col10">0.47</td>
<td class="data row5 col11" id="T_00aa2_row5_col11">0.47</td>
<td class="data row5 col12" id="T_00aa2_row5_col12">0.50</td>
<td class="data row5 col13" id="T_00aa2_row5_col13">0.16</td>
<td class="data row5 col14" id="T_00aa2_row5_col14">0.54</td>
<td class="data row5 col15" id="T_00aa2_row5_col15">0.09</td>
<td class="data row5 col16" id="T_00aa2_row5_col16">0.57</td>
<td class="data row5 col17" id="T_00aa2_row5_col17">0.41</td>
<td class="data row5 col18" id="T_00aa2_row5_col18">0.64</td>
<td class="data row5 col19" id="T_00aa2_row5_col19">0.57</td>
<td class="data row5 col20" id="T_00aa2_row5_col20">0.14</td>
<td class="data row5 col21" id="T_00aa2_row5_col21">nan</td>
<td class="data row5 col22" id="T_00aa2_row5_col22">0.25</td>
</tr>
<tr>
<th class="row_heading level0 row6" id="T_00aa2_level0_row6">otter</th>
<td class="data row6 col0" id="T_00aa2_row6_col0">0.22</td>
<td class="data row6 col1" id="T_00aa2_row6_col1">0.37</td>
<td class="data row6 col2" id="T_00aa2_row6_col2">0.03</td>
<td class="data row6 col3" id="T_00aa2_row6_col3">0.38</td>
<td class="data row6 col4" id="T_00aa2_row6_col4">0.32</td>
<td class="data row6 col5" id="T_00aa2_row6_col5">0.42</td>
<td class="data row6 col6" id="T_00aa2_row6_col6">0.38</td>
<td class="data row6 col7" id="T_00aa2_row6_col7">0.02</td>
<td class="data row6 col8" id="T_00aa2_row6_col8">0.51</td>
<td class="data row6 col9" id="T_00aa2_row6_col9">0.17</td>
<td class="data row6 col10" id="T_00aa2_row6_col10">0.53</td>
<td class="data row6 col11" id="T_00aa2_row6_col11">0.10</td>
<td class="data row6 col12" id="T_00aa2_row6_col12">0.57</td>
<td class="data row6 col13" id="T_00aa2_row6_col13">0.41</td>
<td class="data row6 col14" id="T_00aa2_row6_col14">0.66</td>
<td class="data row6 col15" id="T_00aa2_row6_col15">0.30</td>
<td class="data row6 col16" id="T_00aa2_row6_col16">0.59</td>
<td class="data row6 col17" id="T_00aa2_row6_col17">0.60</td>
<td class="data row6 col18" id="T_00aa2_row6_col18">0.65</td>
<td class="data row6 col19" id="T_00aa2_row6_col19">0.59</td>
<td class="data row6 col20" id="T_00aa2_row6_col20">0.27</td>
<td class="data row6 col21" id="T_00aa2_row6_col21">0.50</td>
<td class="data row6 col22" id="T_00aa2_row6_col22">0.35</td>
</tr>
</tbody>
</table>


### Valid Answer Ratio - Evaluation without Post-Processing Tolerance


<table id="T_d3a51">
<thead>
<tr>
<th class="index_name level0">Dataset</th>
<th class="col_heading level0 col0" colspan="2" id="T_d3a51_level0_col0" style="border-bottom: 1px solid black;">aokvqa</th>
<th class="col_heading level0 col2" id="T_d3a51_level0_col2" style="border-bottom: 1px solid black;">clevr</th>
<th class="col_heading level0 col3" id="T_d3a51_level0_col3" style="border-bottom: 1px solid black;">esnlive</th>
<th class="col_heading level0 col4" id="T_d3a51_level0_col4" style="border-bottom: 1px solid black;">gqa</th>
<th class="col_heading level0 col5" id="T_d3a51_level0_col5" style="border-bottom: 1px solid black;">hateful_memes</th>
<th class="col_heading level0 col6" id="T_d3a51_level0_col6" style="border-bottom: 1px solid black;">mami</th>
<th class="col_heading level0 col7" id="T_d3a51_level0_col7" style="border-bottom: 1px solid black;">mvsa</th>
<th class="col_heading level0 col8" id="T_d3a51_level0_col8" style="border-bottom: 1px solid black;">okvqa</th>
<th class="col_heading level0 col9" id="T_d3a51_level0_col9" style="border-bottom: 1px solid black;">scienceqa</th>
<th class="col_heading level0 col10" id="T_d3a51_level0_col10" style="border-bottom: 1px solid black;">Average Ratio</th>
</tr>
<tr>
<th class="index_name level1">Task</th>
<th class="col_heading level1 col0" id="T_d3a51_level1_col0">direct answer (aokvqa)</th>
<th class="col_heading level1 col1" id="T_d3a51_level1_col1">multiple choice (aokvqa)</th>
<th class="col_heading level1 col2" id="T_d3a51_level1_col2">direct answer (clevr)</th>
<th class="col_heading level1 col3" id="T_d3a51_level1_col3">entailment prediction</th>
<th class="col_heading level1 col4" id="T_d3a51_level1_col4">direct answer (gqa)</th>
<th class="col_heading level1 col5" id="T_d3a51_level1_col5">hate classification</th>
<th class="col_heading level1 col6" id="T_d3a51_level1_col6">sexism classification</th>
<th class="col_heading level1 col7" id="T_d3a51_level1_col7">sentiment analysis</th>
<th class="col_heading level1 col8" id="T_d3a51_level1_col8">direct answer (okvqa)</th>
<th class="col_heading level1 col9" id="T_d3a51_level1_col9">multiple choice (sqa)</th>
<th class="col_heading level1 col10" id="T_d3a51_level1_col10"></th>
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
</tr>
</thead>
<tbody>
<tr>
<th class="row_heading level0 row0" id="T_d3a51_level0_row0">adept</th>
<td class="data row0 col0" id="T_d3a51_row0_col0">1.00</td>
<td class="data row0 col1" id="T_d3a51_row0_col1">0.83</td>
<td class="data row0 col2" id="T_d3a51_row0_col2">1.00</td>
<td class="data row0 col3" id="T_d3a51_row0_col3">0.99</td>
<td class="data row0 col4" id="T_d3a51_row0_col4">1.00</td>
<td class="data row0 col5" id="T_d3a51_row0_col5">0.11</td>
<td class="data row0 col6" id="T_d3a51_row0_col6">0.26</td>
<td class="data row0 col7" id="T_d3a51_row0_col7">0.98</td>
<td class="data row0 col8" id="T_d3a51_row0_col8">1.00</td>
<td class="data row0 col9" id="T_d3a51_row0_col9">0.02</td>
<td class="data row0 col10" id="T_d3a51_row0_col10">0.72</td>
</tr>
<tr>
<th class="row_heading level0 row1" id="T_d3a51_level0_row1">blip2</th>
<td class="data row1 col0" id="T_d3a51_row1_col0">1.00</td>
<td class="data row1 col1" id="T_d3a51_row1_col1">1.00</td>
<td class="data row1 col2" id="T_d3a51_row1_col2">1.00</td>
<td class="data row1 col3" id="T_d3a51_row1_col3">1.00</td>
<td class="data row1 col4" id="T_d3a51_row1_col4">1.00</td>
<td class="data row1 col5" id="T_d3a51_row1_col5">1.00</td>
<td class="data row1 col6" id="T_d3a51_row1_col6">1.00</td>
<td class="data row1 col7" id="T_d3a51_row1_col7">1.00</td>
<td class="data row1 col8" id="T_d3a51_row1_col8">1.00</td>
<td class="data row1 col9" id="T_d3a51_row1_col9">0.88</td>
<td class="data row1 col10" id="T_d3a51_row1_col10">0.99</td>
</tr>
<tr>
<th class="row_heading level0 row2" id="T_d3a51_level0_row2">idefics</th>
<td class="data row2 col0" id="T_d3a51_row2_col0">1.00</td>
<td class="data row2 col1" id="T_d3a51_row2_col1">0.07</td>
<td class="data row2 col2" id="T_d3a51_row2_col2">1.00</td>
<td class="data row2 col3" id="T_d3a51_row2_col3">0.03</td>
<td class="data row2 col4" id="T_d3a51_row2_col4">1.00</td>
<td class="data row2 col5" id="T_d3a51_row2_col5">1.00</td>
<td class="data row2 col6" id="T_d3a51_row2_col6">0.94</td>
<td class="data row2 col7" id="T_d3a51_row2_col7">0.00</td>
<td class="data row2 col8" id="T_d3a51_row2_col8">1.00</td>
<td class="data row2 col9" id="T_d3a51_row2_col9">0.00</td>
<td class="data row2 col10" id="T_d3a51_row2_col10">0.60</td>
</tr>
<tr>
<th class="row_heading level0 row3" id="T_d3a51_level0_row3">instructblip</th>
<td class="data row3 col0" id="T_d3a51_row3_col0">1.00</td>
<td class="data row3 col1" id="T_d3a51_row3_col1">1.00</td>
<td class="data row3 col2" id="T_d3a51_row3_col2">1.00</td>
<td class="data row3 col3" id="T_d3a51_row3_col3">1.00</td>
<td class="data row3 col4" id="T_d3a51_row3_col4">1.00</td>
<td class="data row3 col5" id="T_d3a51_row3_col5">1.00</td>
<td class="data row3 col6" id="T_d3a51_row3_col6">0.93</td>
<td class="data row3 col7" id="T_d3a51_row3_col7">0.99</td>
<td class="data row3 col8" id="T_d3a51_row3_col8">1.00</td>
<td class="data row3 col9" id="T_d3a51_row3_col9">0.04</td>
<td class="data row3 col10" id="T_d3a51_row3_col10">0.90</td>
</tr>
<tr>
<th class="row_heading level0 row4" id="T_d3a51_level0_row4">llava</th>
<td class="data row4 col0" id="T_d3a51_row4_col0">1.00</td>
<td class="data row4 col1" id="T_d3a51_row4_col1">0.00</td>
<td class="data row4 col2" id="T_d3a51_row4_col2">1.00</td>
<td class="data row4 col3" id="T_d3a51_row4_col3">0.00</td>
<td class="data row4 col4" id="T_d3a51_row4_col4">1.00</td>
<td class="data row4 col5" id="T_d3a51_row4_col5">0.00</td>
<td class="data row4 col6" id="T_d3a51_row4_col6">0.00</td>
<td class="data row4 col7" id="T_d3a51_row4_col7">0.00</td>
<td class="data row4 col8" id="T_d3a51_row4_col8">1.00</td>
<td class="data row4 col9" id="T_d3a51_row4_col9">0.00</td>
<td class="data row4 col10" id="T_d3a51_row4_col10">0.40</td>
</tr>
<tr>
<th class="row_heading level0 row5" id="T_d3a51_level0_row5">openflamingo</th>
<td class="data row5 col0" id="T_d3a51_row5_col0">1.00</td>
<td class="data row5 col1" id="T_d3a51_row5_col1">0.09</td>
<td class="data row5 col2" id="T_d3a51_row5_col2">1.00</td>
<td class="data row5 col3" id="T_d3a51_row5_col3">0.20</td>
<td class="data row5 col4" id="T_d3a51_row5_col4">1.00</td>
<td class="data row5 col5" id="T_d3a51_row5_col5">0.38</td>
<td class="data row5 col6" id="T_d3a51_row5_col6">0.69</td>
<td class="data row5 col7" id="T_d3a51_row5_col7">0.80</td>
<td class="data row5 col8" id="T_d3a51_row5_col8">1.00</td>
<td class="data row5 col9" id="T_d3a51_row5_col9">0.00</td>
<td class="data row5 col10" id="T_d3a51_row5_col10">0.62</td>
</tr>
<tr>
<th class="row_heading level0 row6" id="T_d3a51_level0_row6">otter</th>
<td class="data row6 col0" id="T_d3a51_row6_col0">1.00</td>
<td class="data row6 col1" id="T_d3a51_row6_col1">0.66</td>
<td class="data row6 col2" id="T_d3a51_row6_col2">1.00</td>
<td class="data row6 col3" id="T_d3a51_row6_col3">0.62</td>
<td class="data row6 col4" id="T_d3a51_row6_col4">1.00</td>
<td class="data row6 col5" id="T_d3a51_row6_col5">0.91</td>
<td class="data row6 col6" id="T_d3a51_row6_col6">0.74</td>
<td class="data row6 col7" id="T_d3a51_row6_col7">0.97</td>
<td class="data row6 col8" id="T_d3a51_row6_col8">1.00</td>
<td class="data row6 col9" id="T_d3a51_row6_col9">0.01</td>
<td class="data row6 col10" id="T_d3a51_row6_col10">0.79</td>
</tr>
</tbody>
</table>

## Model Implementation

For some models, the respective GitHub repository needed to be cloned and files tweaked. 
The cloned and modified repositories are collected in `models`. 
To implement the models yourself, follow the instructions in `MAKE_ME_RUN.md`, 
that can be found in `models/model_of_interest`.

