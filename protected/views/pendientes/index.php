<?php
$this->breadcrumbs = array(
	'Pendientes',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' Pendientes', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' Pendientes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Pendientes'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
