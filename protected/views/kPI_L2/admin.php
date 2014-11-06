<?php
$this->breadcrumbs=array(
	'Kpi  L2s'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Manage'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'List KPI_L2'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Create KPI_L2'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('kpi--l2-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Manage&nbsp;Kpi  L2s'); ?>

<?php echo CHtml::link(Yii::t('app', 'Advanced Search'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'kpi--l2-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
	'columns'=>array(
		'id',
		'Fecha',
		'Eff_Shift',
		'Count_Fill_Batch',
		'Count_Pall_Batch',
		'Count_Fill_Shift',
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
