
<?php
    $sufix = "";
if (isset($_GET['query'])) {
        $sufix = "?query=" . urlencode($_GET['query']);
}
?>

<?php
    
$this->breadcrumbs=array(
	'Documentos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);
         $this->menu = array(
             array('label'=>Yii::t('app',
				'Lista de Documentos'), 'url'=>array('index')),
            array('label' => 'Subir Doc. de Motor', 'url' => array('/Documentos/createSubirMotor'.$sufix)),
            array('label' => 'Subir Doc. de Equipo', 'url' => array('/Documentos/createSubirEquipo'.$sufix)),
            array('label' => 'Subir Doc. de Tablero', 'url' => array('/Documentos/createSubirTablero'.$sufix)),
        );



		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('meta-docs-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle ('Gestionar&nbsp;Documentos'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'/index.php/metaDocs/admin?avanzada=1',array('class'=>'search-button')); ?>
<?php
    if (isset($_GET['avanzada']))
        echo'<div class="search-form" >';
    else
        echo'<div class="search-form" style="display:none">';
?>
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'meta-docs-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
         'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
	'columns'=>array(
		//'id',
		'titulo',
                array(
                    'header'=>'Medio',
                    'type'=>'raw',
                    'value'=>'CHtml::link($data->medio0->descripcion,"/index.php/metaDocs/".$data->id)',
                    //'value'=>'CHtml::label( $data->id,false)'
                ),
                'tipoContenido0.descripcion',
		'fabricante0.descripcion',
		//'cerveceria',
		//'numPedido',
		//'numComision',
		'ubicacionT0.descripcion',
		//'descripcion',
		
		//'version',

		//'idioma0',
		'disponibles',
		'existencias',
		/*
                'modulo',
		'columna',
		'fila',
		'documento',
		'ruta',
		'fechaCreacion',
		'fechaRecepcion',
		'autores',
		'usuario',
		'revisado',
		'userRevisado',
		'fechaRevisado',
		'eliminado',
		'secuencia',
		'ordenSecuencia',
                'ISBN',
		'EAN13',
		*/
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
