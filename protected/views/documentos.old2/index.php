<?php
$this->breadcrumbs = array(
	'Unidades Documentales',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Nueva Unidad') , 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar Unidades') , 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle('Unidades Documentales:'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
