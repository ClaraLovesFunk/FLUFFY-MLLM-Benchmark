# Testing-Multimodal-LLMs

## Benchmark


<table id="T_61b5e">
<thead>
<tr>
<th class="index_name level0">Dataset</th>
<th class="col_heading level0 col0" colspan="2" id="T_61b5e_level0_col0" style="border-bottom: 1px solid black;">aokvqa</th>
<th class="col_heading level0 col2" id="T_61b5e_level0_col2" style="border-bottom: 1px solid black;">okvqa</th>
</tr>
<tr>
<th class="index_name level1">Task</th>
<th class="col_heading level1 col0" id="T_61b5e_level1_col0">direct answer</th>
<th class="col_heading level1 col1" id="T_61b5e_level1_col1">multiple choice</th>
<th class="col_heading level1 col2" id="T_61b5e_level1_col2">direct answer</th>
</tr>
<tr>
<th class="index_name level2">Metric</th>
<th class="col_heading level2 col0" id="T_61b5e_level2_col0">acc</th>
<th class="col_heading level2 col1" id="T_61b5e_level2_col1">acc</th>
<th class="col_heading level2 col2" id="T_61b5e_level2_col2">acc</th>
</tr>
<tr>
<th class="index_name level0"></th>
<th class="blank col0"> </th>
<th class="blank col1"> </th>
<th class="blank col2"> </th>
</tr>
</thead>
<tbody>
<tr>
<th class="row_heading level0 row0" id="T_61b5e_level0_row0">blip2</th>
<td class="data row0 col0" id="T_61b5e_row0_col0">0.24</td>
<td class="data row0 col1" id="T_61b5e_row0_col1">0.70</td>
<td class="data row0 col2" id="T_61b5e_row0_col2">29.53</td>
</tr>
</tbody>
</table>
   

## Checklist




| Models              | A-OKVQA | OKVQA | VQA-v2 | EMU | E-SNLI-VE | VCR |
|---------------------|---------|-------|--------|-----|-----------|-----|
| BLIP-2              |&#x2714; &#x2714;|&#x2714; &#x2714;|        |     |           |     |
| BLIP-vicuna         |         |       |        |     |           |     |
| Prismer             |         |       |        |     |           |     |
| OpenFlamingo*       |         |       |        |     |           |     |
| MiniGPT             |         |       |        |     |           |     |
| Llava               |         |       |        |     |           |     |
| Otter*              |         |       |        |     |           |     |
| Fromage             |         |       |        |     |           |     |
| MAGMA (no hf)       |         |       |        |     |           |     |
| Limber (no hf)      |         |       |        |     |           |     |
| MAPL (no hf)        |         |       |        |     |           |     |
| FLAN-T5 (text)      |         |       |        |     |           |     |
| GPT4AIl (text)      |         |       |        |     |           |     |
| OpenAssistant (text)|         |       |        |     |           |     |
* Paper Coming Soon
