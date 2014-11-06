<?php
$this->breadcrumbs = array(
	'Kpi  L1s',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' KPI_L1', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' KPI_L1', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Kpi  L1s'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
