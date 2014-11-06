<?php
$this->breadcrumbs=array(
	'Analistas'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Analistas'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Crear Analistas'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('analistas-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Analistas'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'analistas-grid',
	'dataProvider'=>$model->search(),
	//'filter'=>$model,
        'cssFile'=>'/themes/gridview/styles.css',
	'columns'=>array(
		'id',
		'Analista',
		'Proceso',
		'Pto_trabajo',
		'modulo',
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
