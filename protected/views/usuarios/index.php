<?php
$this->breadcrumbs = array(
	'Usuarios',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Usuarios', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Usuarios', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Usuarios'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
