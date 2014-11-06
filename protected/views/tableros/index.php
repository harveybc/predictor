<?php
$this->breadcrumbs = array(
	'Tableros',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	//array('label'=>Yii::t('app', 'Crear') . ' Tableros', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Tableros', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Tableros'); ?>
<div class="forms100c" style="text-align:left;">
<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
</div>