<?php
$this->breadcrumbs=array(
	'Unidades Documentales'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Unidades'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Nueva Unidad'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('documentos-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Unidades Documentales'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'documentos-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
        'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
	'columns'=>array(
		//'id',
		'descripcion',
		
		/*
                 * 'permitirAdiciones',
		'permitirAnotaciones',
		'autorizarOtros',
		'requiereAutorizacion',
		'secuencia',
		'ordenSecuencia',
		'eliminado',
		'conservacionInicio',
		'conservacionFin',
		'conservacionPermanente',
		
		*/
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
