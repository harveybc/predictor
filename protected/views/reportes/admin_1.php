<?php
$this->breadcrumbs=array(
	'Reportes'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Reportes'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Nuevo Reporte'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('reportes-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Reportes'); ?>



<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'reportes-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
	'columns'=>array(
		'id',
		'Reporte',
		'Path',
		'Presion',
		'Decibeles',
		'Descripcion',
		/*
		'ZI',
		'Proceso',
		'Area',
		'Equipo',
		'Analista',
		'OT',
		'Fecha',
		'Gas',
		'Tamano',
		'CFM',
		'COSTO',
		'Corregido',
		*/
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
