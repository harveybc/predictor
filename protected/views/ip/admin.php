<?php
$this->breadcrumbs=array(
	'Ips'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'List Ip'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Create Ip'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('ip-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Ips'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'ip-grid',
	'dataProvider'=>$model->search(),
	//'filter'=>$model,
        'cssFile'=>'/themes/gridview/styles.css',
        'cssFile'=>'/themes/gridview/styles.css',
	'columns'=>array(
		'id',
		'Fecha',
		'TAG',
		'IP',
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
