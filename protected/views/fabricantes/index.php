<?php
$this->breadcrumbs = array(
	'Fabricantes',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Fabricantes', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Fabricantes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Fabricantes'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
