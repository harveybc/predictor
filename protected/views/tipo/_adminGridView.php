<?php
$model = new Tipo('search');
$this->widget('zii.widgets.grid.CGridView', array(
    'id' => 'tipo-grid',
    'dataProvider' => Tipo::model()->searchFechas($proceso),
   // 'filter' => $model,
    'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
    'columns' => array(
        	//'id',
        	'Tipo_Aceite',
                'Proceso',
                 array(
            'class' => 'CButtonColumn',
        ),
    ),
));
?>
<!--TODO:falta arreglar numero de paginas para que no se vean supe rpuestas --->