<?php
$this->breadcrumbs=array(
	'documentos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de documentos'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Nuevo Metadocumento'),
				'url'=>array('create')),
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

<?php $this->setPageTitle ('Gestionar&nbsp;documentos'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
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
		'tipoContenido',
		'fabricante',
		'cerveceria',
		'numPedido',
		'numComision',
		/*
		'ubicacionT',
		'descripcion',
		'titulo',
		'version',
		'medio',
		'idioma',
		'disponibles',
		'existencias',
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
