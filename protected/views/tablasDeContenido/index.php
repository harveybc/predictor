<?php
$this->breadcrumbs = array(
	'Tablas De Contenidos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' TablasDeContenido', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' TablasDeContenido', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Tablas De Contenidos'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
