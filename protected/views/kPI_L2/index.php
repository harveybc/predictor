<?php
$this->breadcrumbs = array(
	'Kpi  L2s',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' KPI_L2', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' KPI_L2', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Kpi  L2s'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
