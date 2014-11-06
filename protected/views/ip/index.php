<?php
$this->breadcrumbs = array(
	'Ips',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Ip', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Ip', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Ips'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
