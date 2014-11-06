<?php
$this->breadcrumbs = array(
	'Tags',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' Tags', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' Tags', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Tags'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
