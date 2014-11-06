<?php
$this->breadcrumbs = array(
	'Cervecerias',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Cervecerias', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Cervecerias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Cervecerias'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
