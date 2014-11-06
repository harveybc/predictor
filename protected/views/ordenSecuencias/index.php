<?php
$this->breadcrumbs = array(
	'Orden Secuenciases',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' OrdenSecuencias', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' OrdenSecuencias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Orden Secuenciases'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
