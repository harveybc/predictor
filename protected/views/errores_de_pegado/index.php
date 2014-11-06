<?php
$this->breadcrumbs = array(
	'Errores De Pegados',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' Errores_de_pegado', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' Errores_de_pegado', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Errores De Pegados'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
