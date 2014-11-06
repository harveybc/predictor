<?php
$this->breadcrumbs = array(
	'Anotaciones',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Nueva') . ' Anotación', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' Anotaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle('Anotaciones'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
