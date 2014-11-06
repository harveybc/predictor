<?php
$this->breadcrumbs = array(
	'Lubricantes',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Lubricante', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Lubricantes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Lubricantes'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
